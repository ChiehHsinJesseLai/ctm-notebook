"""
Train a diffusion model on images.
"""

import argparse

# from cm import logger

import torch as th
import blobfile as bf
import random

from PIL import Image
import blobfile as bf

import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import timm
import math
import torch.nn.functional as F
import types
# from pg_modules.blocks import FeatureFusionBlock
from torch.nn.utils import spectral_norm
from scipy.stats import norm
import sys
import copy
import glob
import scipy
import inspect
import io
import pickle
# import dnnlib
import uuid
from torch.nn.functional import silu
import torchvision
from torchvision.transforms import Normalize
from torchvision.utils import make_grid, save_image
import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union, Optional
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import re
import requests
import hashlib


_decorators = set()
_module_to_src_dict = dict()
_src_to_module_dict = dict()
_import_hooks = [] 
_version = 6
_dnnlib_cache_dir = None


# Modules of dnnlib are here

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        safe_name = safe_name[:min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)

def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """Determine whether the given object is a valid URL string."""
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True

def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)

class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, lowest=False):
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output = self.skip_add.add(output, xs[1])

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

def ctm_data_defaults(data_name):
    return dict(
        train_classes=-1,
        type='png',
        sigma_data=0.5,
        deterministic=False,
        num_classes=10,
    )

def ctm_loss_defaults(data_name):
    return dict(
        # CTM hyperparams
        ctm_training=True,
        consistency_weight=1.0,
        ctm_estimate_outer_type='target_model_sg',
        ctm_estimate_inner_type='model',
        ctm_target_inner_type='model_sg',
        ctm_target_matching=False,
        sample_s_strategy='uniform',
        heun_step_strategy='weighted',
        heun_step_multiplier=1.0,
        outer_parametrization='euler',
        inner_parametrization='edm',
        time_continuous=False,
        self_learn=False,
        self_learn_iterative=False,
        target_matching=False,

        # DSM hyperparams
        diffusion_training=True,
        apply_adaptive_weight=True,
        denoising_weight=1.,
        diffusion_mult = 0.7,
        diffusion_schedule_sampler='halflognormal',
        diffusion_training_frequency=1.,

        # GAN hyperparams
        d_lr=0.002,
        gan_training=False,
        gan_specific_batch=False,
        gan_micro_batch=32,
        gan_real_free=True,
        discriminator_weight=1.0,
        discriminator_start_itr=0,
        use_d_fp16=False,
        d_architecture='StyleGAN-XL',
        g_learning_period=1,
        gan_fake_outer_type='no',
        gan_fake_inner_type='',
        gan_real_inner_type='',
        gan_target_matching=False,
        data_augment=True,
        d_backbone=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'],
        d_apply_adaptive_weight=True,
        shift_ratio=0.125,
        cutout_ratio=0.2,
        gan_training_frequency=1.,
        gaussian_filter=False,
        blur_fade_itr=1000,
        blur_init_sigma=2,
        prob_aug=1.0,
        gan_different_augment=False,
        gan_num_heun_step=17 if data_name == 'cifar10' else 39,
        gan_heun_step_strategy='uniform',
        gan_specific_time=False,
        gan_low_res_train=False,
        d_opt_load=True,
    )

def ctm_train_defaults(data_name):
    return dict(
        beta_min=0.1,
        beta_max=20.,
        multiplier=1.,
        num_heun_step=17 if data_name == 'cifar10' else 39,
        num_heun_step_random=True,

        # Network architecture
        edm_nn_ncsn=False,
        edm_nn_ddpm=True if data_name == 'cifar10' else False,
        in_channels=3,
        linear_probing=False,
        target_subtract=False,
    )

def ctm_eval_defaults(data_name):
    return dict(
        intermediate_samples=False,
        sampling_batch=64,
        sample_interval=10 if data_name == 'cifar10' else 10,
        sampling_steps=18 if data_name == 'cifar10' else 40,
        eval_interval=10,
        eval_num_samples=5,
        eval_batch=500,
        #ref_path='/home/dongjun/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz' if data_name == 'cifar10' else "",
        ref_path='/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz' if data_name == 'cifar10' \
            else "/home/fp084243/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz",
        ref_feat_path='',
        large_log=False,
        compute_ema_fids=False,
        #dm_sample_path_seed_42='/data2/dongjun/EighthArticleExperimentalResults/CIFAR10/DM/EDM-VP/fp16-seed-42/edm_heun_sampler_18_steps_ond-vp_itrs_model_ema' if data_name == 'cifar10' else "",
        dm_sample_path_seed_42='/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/DM/heun_18_seed_42_ver2' if data_name == 'cifar10' else "",
        ae_image_path_seed_42='',
        eval_seed=42,
        eval_fid=False,
        eval_similarity=True,
        save_png=False,
        check_ctm_denoising_ability=False,
        check_dm_performance=True,
        sanity_check=False,
        save_period=10 if data_name == 'cifar10' else 10,
        clip_denoised=False,
        clip_output=True,
        gpu_usage=False,
        eval_large_nfe=True,
    )

def cm_train_defaults(data_name):
    return dict(
        #teacher_model_path="/home/dongjun/EighthArticleExperimentalResults/CIFAR10/author_ckpt/edm-cifar10-32x32-uncond-vp.pkl" if data_name == 'cifar10' else "",
        teacher_model_path="/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/edm-cifar10-32x32-uncond-vp.pkl" if data_name == 'cifar10' else "",
        teacher_dropout=0.0 if data_name == 'cifar10' else 0.1,
        training_mode="ctm",
        target_ema_mode="fixed",
        scale_mode="fixed",
        total_training_steps=100,
        start_ema=0.999,
        start_scales=18 if data_name == 'cifar10' else 40,
        end_scales=18 if data_name == 'cifar10' else 40,
        distill_steps_per_iter=50,
        loss_norm="lpips",
        port=6,
    )

def model_and_diffusion_defaults(data_name):
    """
    Defaults for image training.
    """
    res = dict(
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7,
        image_size=32 if data_name == 'cifar10' else 64,
        num_channels=192,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False if data_name == 'cifar10' else True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        learn_sigma=False,
        weight_schedule="uniform",
        weight_schedule_multiplier=1.,
        diffusion_weight_schedule="karras_weight",
        rescaling=False,
    )
    return res

def train_defaults(data_name):
    """
        Defaults for model training.
    """
    res = dict(
        out_dir="",
        #data_dir="/home/dongjun/EighthArticleExperimentalResults/CIFAR10/train" if data_name == 'cifar10' else "",
        data_dir="/home/acf15618av/dataset/CIFAR10/train" if data_name == 'cifar10' else "",
        schedule_sampler="uniform",
        lr=0.0004 if data_name == 'cifar10' else 0.000008,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=128 if data_name == 'cifar10' else 2048,
        batch_size=-1,
        microbatch=64 if data_name.lower() == 'cifar10' else -1,  # -1 disables microbatches
        ema_rate="0.999,0.9999" if data_name == 'cifar10' else "0.999,0.9999,0.9999432189950708",
        # comma-separated list of EMA values
        log_interval=10,
        save_interval=100,
        save_check_period=100,
        resume_checkpoint="",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        device_id=0,
        num_workers=16,
        use_MPI=False,
        map_location='cuda',
    )
    return res

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def load_data(
    *,
    args,
    data_dir,
    batch_size,
    image_size,
    data_name='cifar10',
    random_crop=False,
    random_flip=True,
    num_workers=32,
    type='jpeg',
    flip_ratio=0.5,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    all_files = _list_image_files_recursively(data_dir)
    classes = None



    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        random_crop=random_crop,
        random_flip=random_flip,
        data_name=data_name,
        type=type,
        flip_ratio=flip_ratio,
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    while True:
        yield from loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy", "jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        random_crop=False,
        random_flip=True,
        data_name='cifar10',
        type='jpeg',
        flip_ratio=0.5,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths#[shard:][::num_shards]
        print(f"Total number of data: {len(image_paths)}, data for shard/num_shards device: {len(self.local_images)}")
        self.local_classes = None if classes is None else classes#[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.data_name = data_name
        self.type = type
        self.flip_ratio = flip_ratio

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        # if self.type == 'npy':
        #     print("i am inside self.tupe == npy")
        #     exit()
        #     data = np.load(path)
        #     out_dict = {}
        #     if self.local_classes is not None:
        #         print("i am inside self.local_classes is not None")
        #         out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        #     return data, out_dict
        # else:
        # print("I am inside else loop")
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < self.flip_ratio:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        return np.transpose(arr, [2, 0, 1]), out_dict

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file for single GPU.
    """
    # Directly load the state dictionary from the file
    state_dict = th.load(path, **kwargs)
    return state_dict

def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            print('Printing from the if block of ema and scales fn')
            target_ema = start_ema
            scales = start_scales
        else:
            print("check if other options are commented out")
            print("printing the target_ema_mode:: ", target_ema_mode)
            print("scale_mode:: ", scale_mode)
            raise NotImplementedError

        return float(target_ema), int(scales)

    return ema_and_scales_fn

def load_feature_extractor(args, eval=True):
    feature_extractor = None
    if args.loss_norm == 'lpips':
        from piq import LPIPS
        feature_extractor = LPIPS(replace_pooling=True, reduction="none")
    return feature_extractor

# for GAN projector
class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]
    
class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)

class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = th.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x.contiguous()

def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = th.cat([posemb_tok, posemb_grid], dim=1)

    return posemb

def forward_flex(self, x):
    b, c, h, w = x.shape

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    if hasattr(self, "dist_token") and self.dist_token is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = th.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = th.cat((cls_tokens, x), dim=1)

    x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)

    return x

def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook

def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    # 32, 48, 136, 384
    pretrained.layer1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, th.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.layer2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, th.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.layer3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, th.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.layer4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, th.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained

def calc_dims(pretrained, is_vit=False):
    dims = []
    inp_res = 256
    tmp = th.zeros(1, 3, inp_res, inp_res)
    if not is_vit:
        tmp = pretrained.layer0(tmp)
        dims.append(tmp.shape[1:3])
        tmp = pretrained.layer1(tmp)
        dims.append(tmp.shape[1:3])
        tmp = pretrained.layer2(tmp)
        dims.append(tmp.shape[1:3])
        tmp = pretrained.layer3(tmp)
        dims.append(tmp.shape[1:3])
    else:
        tmp = forward_vit(pretrained, tmp)
        dims = [out.shape[1:3] for out in tmp]

    # split to channels and resolution multiplier
    dims = np.array(dims)
    channels = dims[:, 0]
    res_mult = dims[:, 1] / inp_res
    return channels, res_mult

def _make_vit(model, name):
    # print(name)
    if 'base' in name:
        print(" i am inside the loop")
        features = [96, 192, 384, 768]
        hooks = [2, 5, 8, 11]
        vit_features = 768
    else:
        print("NOPR")
    return _make_vit_b16_backbone(
    model,
    features=features,
    size=[224, 224],
    hooks=hooks,
    vit_features=vit_features,
    start_index=2 if 'deit' in name else 1,
    )

def _make_efficientnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(
        model.conv_stem, model.bn1, model.act1, *model.blocks[0:2]
    )
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained

def _make_pretrained(backbone, verbose=False):
    print("printing the name of backbone::::: ", backbone)

    if backbone == "deit_base_distilled_patch16_224":
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_vit(model, backbone)
        is_vit = True
    elif backbone == "tf_efficientnet_lite0":
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)
        is_vit = False
    pretrained.CHANNELS, pretrained.RES_MULT = calc_dims(pretrained, is_vit)
    return pretrained

def get_backbone_normstats(backbone):
    if backbone :
        return {
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
        }
    else:
        print("check if the necesarry blocks are delted. it is in pg_modules.projector.py")
        raise NotImplementedError

def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    # shapes
    out_channels = [cout, cout*2, cout*4, cout*8] if expand else [cout]*4

    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)

    scratch.CHANNELS = out_channels

    return scratch

def _make_scratch_csm(scratch, in_channels, cout, expand):
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], nn.ReLU(False), expand=expand, lowest=True)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], nn.ReLU(False), expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], nn.ReLU(False), expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0], nn.ReLU(False))

    # last refinenet does not expand to save channels in higher dimensions
    scratch.CHANNELS = [cout, cout, cout*2, cout*4] if expand else [cout]*4

    return scratch

def _make_projector(im_res, backbone, cout, proj_type, expand=False):
    assert proj_type in [0, 1, 2], "Invalid projection type"

    ### Build pretrained feature network
    pretrained = _make_pretrained(backbone)

    # Following Projected GAN
    im_res = 256
    pretrained.RESOLUTIONS = [im_res//4, im_res//8, im_res//16, im_res//32]

    if proj_type == 0: return pretrained, None
    # print(pretrained.CHANNELS)
    # print(pretrained)
    # exit()

    ### Build CCM
    scratch = nn.Module()
    scratch = _make_scratch_ccm(scratch, in_channels=pretrained.CHANNELS, cout=cout, expand=expand)

    pretrained.CHANNELS = scratch.CHANNELS

    if proj_type == 1: return pretrained, scratch

    ### build CSM
    scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS, cout=cout, expand=expand)

    # CSM upsamples x2 so the feature map resolution doubles
    pretrained.RESOLUTIONS = [res*2 for res in pretrained.RESOLUTIONS]
    pretrained.CHANNELS = scratch.CHANNELS

    return pretrained, scratch

def forward_vit(pretrained, x):
    b, c, h, w = x.shape
    _ = pretrained.model.forward_flex(x)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    layer_1 = pretrained.layer1[0:2](layer_1)
    layer_2 = pretrained.layer2[0:2](layer_2)
    layer_3 = pretrained.layer3[0:2](layer_3)
    layer_4 = pretrained.layer4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            th.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.layer1[3 : len(pretrained.layer1)](layer_1)
    layer_2 = pretrained.layer2[3 : len(pretrained.layer2)](layer_2)
    layer_3 = pretrained.layer3[3 : len(pretrained.layer3)](layer_3)
    layer_4 = pretrained.layer4[3 : len(pretrained.layer4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4

class F_Identity(nn.Module):
    def forward(self, x):
        return x

class F_RandomProj(nn.Module):
    def __init__(
        self,
        backbone="tf_efficientnet_lite3",
        im_res=256,
        cout=64,
        expand=True,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        **kwargs,
    ):
        super().__init__()
        self.proj_type = proj_type
        self.backbone = backbone
        self.cout = cout
        self.expand = expand
        self.normstats = get_backbone_normstats(backbone)

        # build pretrained feature network and random decoder (scratch)
        self.pretrained, self.scratch = _make_projector(im_res=im_res, backbone=self.backbone, cout=self.cout,
                                                        proj_type=self.proj_type, expand=self.expand)
        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS

    def forward(self, x):
        # predict feature maps
        if self.backbone == "deit_base_distilled_patch16_224":
        # if self.backbone:
            out0, out1, out2, out3 = forward_vit(self.pretrained, x)
        # else:
        elif self.backbone == "tf_efficientnet_lite0":

            out0 = self.pretrained.layer0(x)
            out1 = self.pretrained.layer1(out0)
            out2 = self.pretrained.layer2(out1)
            out3 = self.pretrained.layer3(out2)

        # start enumerating at the lowest layer (this is where we put the first discriminator)
        out = {
            '0': out0,
            '1': out1,
            '2': out2,
            '3': out3,
        }

        if self.proj_type == 0: return out

        out0_channel_mixed = self.scratch.layer0_ccm(out['0'])
        out1_channel_mixed = self.scratch.layer1_ccm(out['1'])
        out2_channel_mixed = self.scratch.layer2_ccm(out['2'])
        out3_channel_mixed = self.scratch.layer3_ccm(out['3'])

        out = {
            '0': out0_channel_mixed,
            '1': out1_channel_mixed,
            '2': out2_channel_mixed,
            '3': out3_channel_mixed,
        }

        if self.proj_type == 1: return out

        # from bottom to top
        out3_scale_mixed = self.scratch.layer3_csm(out3_channel_mixed)
        out2_scale_mixed = self.scratch.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.scratch.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.scratch.layer0_csm(out1_scale_mixed, out0_channel_mixed)

        out = {
            '0': out0_scale_mixed,
            '1': out1_scale_mixed,
            '2': out2_scale_mixed,
            '3': out3_scale_mixed,
        }

        return out

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def NormLayer(c, mode='batch'):
    if mode == 'group':
        return nn.GroupNorm(c//2, c)
    elif mode == 'batch':
        return nn.BatchNorm2d(c)

class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, width=1):
        super().__init__()
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes*width, 4, 2, 1, bias=True),
            NormLayer(out_planes*width),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)

class DownBlockPatch(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.main = nn.Sequential(
            DownBlock(in_planes, out_planes),
            conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            NormLayer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)

class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, patch=False):
        super().__init__()

        # midas channels
        nfc_midas = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                     256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in nfc_midas.keys():
            sizes = np.array(list(nfc_midas.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = nfc_midas
        else:
            nfc = {k: ndf for k, v in nfc_midas.items()}

        # for feature map discriminators with nfc not in nfc_midas
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = DownBlockPatch if patch else DownBlock
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz], nfc[start_sz//2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        return self.main(x)

class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=4,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        patch=False,
        **kwargs,
    ):
        super().__init__()

        assert num_discs in [1, 2, 3, 4, 5]

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, patch=patch)],

        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c, rec=False):
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k], c).view(features[k].size(0), -1))

        all_logits = th.cat(all_logits, dim=1)
        return all_logits

def load_discriminator_and_d_feature_extractor(args):
    #assert (args.gan_training == True) == (args.d_architecture == 'StyleGAN-XL')
    if args.gan_training:
        # from pg_modules.projector import F_RandomProj
        # from pg_modules.discriminator import MultiScaleD
        backbones = ['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0']
        discriminator, discriminator_feature_extractor = [], []
        if args.gan_low_res_train:
            discriminator2 = []
        backbone_kwargs = {'im_res': args.image_size}
        for i, bb_name in enumerate(backbones):
            feat = F_RandomProj(bb_name, **backbone_kwargs)
            disc = MultiScaleD(
                channels=feat.CHANNELS,
                resolutions=feat.RESOLUTIONS,
                **backbone_kwargs,
            )
            discriminator_feature_extractor.append([bb_name, feat])
            discriminator.append([bb_name, disc])
            if args.gan_low_res_train:
                discriminator2.append([bb_name + '_low', disc])
        discriminator_feature_extractor = nn.ModuleDict(discriminator_feature_extractor)
        discriminator_feature_extractor = discriminator_feature_extractor.train(False).to(dev)
        discriminator_feature_extractor.requires_grad_(False)
        if args.gan_low_res_train:
            discriminator = discriminator + discriminator2
        discriminator = nn.ModuleDict(discriminator)
        discriminator.to(dev)
        discriminator.train()
        if args.use_d_fp16:
            discriminator.convert_to_fp16()
    else:
        discriminator, discriminator_feature_extractor = None, None
    return discriminator, discriminator_feature_extractor

### GAN part ends here

### Creating model and diffusion

class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self, num_heun_step):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample_t(self, batch_size, device, num_heun_step=1, time_continuous=False):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        if time_continuous:
            indices_np = np.random.rand(batch_size)
            indices = th.from_numpy(indices_np).to(device) * (1. - num_heun_step)
            weights = th.ones_like(indices).float().to(device)
        else:
            w = self.weights(num_heun_step)
            p = w / np.sum(w)
            indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
            indices = th.from_numpy(indices_np).long().to(device)
            weights_np = 1 / (len(p) * p[indices_np])
            weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights

    def sample_s(self, args, batch_size, device, indices, num_heun_step=1, time_continuous=False, N=40):
        if time_continuous:
            new_indices = th.from_numpy(np.random.rand(indices.shape[0])).to(indices.device) * (1. - indices - num_heun_step)\
                          + indices + num_heun_step
        else:
            if args.sample_s_strategy == 'smallest':
                new_indices = th.ones(indices.shape[0], device=indices.device) * (N - 1)
            elif args.sample_s_strategy == 'uniform':
                new_indices = th.from_numpy(np.random.randint(
                    low=(indices + num_heun_step).cpu().detach().numpy(), high=N,
                    size=(indices.shape[0],),
                    dtype=int)).to(indices.device)
        return new_indices

class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self._weights = np.ones([num_timesteps])
        self.num_timesteps = num_timesteps

    def weights(self, num_heun_step=1):
        #if num_heun_step == 1:
        #    return self._weights
        #else:
        return np.ones([self.num_timesteps - num_heun_step])

class HalfLogNormalHalfUniformSampler:
    def __init__(self, args, p_mean=-1.2, p_std=1.2, even=False):
        self.args = args
        self.p_mean = p_mean
        self.p_std = p_std
        self.even = even
        if self.even:
            self.inv_cdf = lambda x: norm.ppf(x, loc=p_mean, scale=p_std)
            # self.rank, self.size = dist.get_rank(), dist.get_world_size()

    def get_t(self, t):
        t = self.args.sigma_max ** (1 / self.args.rho) + t * (
                self.args.sigma_min ** (1 / self.args.rho) - self.args.sigma_max ** (1 / self.args.rho)
        )
        t = t ** self.args.rho
        return t

    def sample(self, bs, device, num_heun_step=1):
        log_sigmas = self.p_mean + self.p_std * th.randn(bs//2, device=device)
        sigmas = th.exp(log_sigmas)
        t = th.rand(bs - bs//2, device=device) * self.args.diffusion_mult # it determines how large sigma to sample from
        sigmas = th.cat((sigmas, self.get_t(t))).view(-1)
        weights = th.ones_like(sigmas)
        return sigmas, weights

def create_named_schedule_sampler(args, name, num_timesteps):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(num_timesteps)
    elif name == 'halflognormal':
        return HalfLogNormalHalfUniformSampler(args)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def mean_flat(tensor, percent='1-'):
    """
    Take the mean over all non-batch dimensions.
    """
    if percent not in ['1-', '1.0-']:
        up = percent[-1]
        percent = float(percent[:-1])
        value = th.quantile(tensor[:64].float(), percent)
        if up == '+':
            tensor = th.where(tensor > value, tensor, 0.)
        elif up == '-':
            tensor = th.where(tensor < value, tensor, 0.)
        else:
            raise NotImplementedError
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def _src_to_module(src):
    r"""Get or create a Python module for the given source code.
    """
    module = _src_to_module_dict.get(src, None)
    if module is None:
        module_name = "_imported_module_" + uuid.uuid4().hex
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        _module_to_src_dict[module] = src
        _src_to_module_dict[src] = module
        exec(src, module.__dict__) # pylint: disable=exec-used
    return module

def _reconstruct_persistent_obj(meta):
    r"""Hook that is called internally by the `pickle` module to unpickle
    a persistent object.
    """
    meta = EasyDict(meta)
    meta.state = EasyDict(meta.state)
    for hook in _import_hooks:
        meta = hook(meta)
        assert meta is not None

    assert meta.version == _version
    module = _src_to_module(meta.module_src)

    assert meta.type == 'class'
    orig_class = module.__dict__[meta.class_name]
    decorator_class = persistent_class(orig_class)
    obj = decorator_class.__new__(decorator_class)

    setstate = getattr(obj, '__setstate__', None)
    if callable(setstate):
        setstate(meta.state) # pylint: disable=not-callable
    else:
        obj.__dict__.update(meta.state)
    return obj

def _check_pickleable(obj):
    r"""Check that the given object is pickleable, raising an exception if
    it is not. This function is expected to be considerably more efficient
    than actually pickling the object.
    """
    def recurse(obj):
        if isinstance(obj, (list, tuple, set)):
            return [recurse(x) for x in obj]
        if isinstance(obj, dict):
            return [[recurse(x), recurse(y)] for x, y in obj.items()]
        if isinstance(obj, (str, int, float, bool, bytes, bytearray)):
            return None # Python primitive types are pickleable.
        if f'{type(obj).__module__}.{type(obj).__name__}' in ['numpy.ndarray', 'torch.Tensor', 'torch.nn.parameter.Parameter']:
            return None # NumPy arrays and PyTorch tensors are pickleable.
        if is_persistent(obj):
            return None # Persistent objects are pickleable, by virtue of the constructor check.
        return obj
    with io.BytesIO() as f:
        pickle.dump(recurse(obj), f)

def _module_to_src(module):
    r"""Query the source code of a given Python module.
    """
    src = _module_to_src_dict.get(module, None)
    if src is None:
        src = inspect.getsource(module)
        _module_to_src_dict[module] = src
        _src_to_module_dict[src] = module
    return src

def is_persistent(obj):
    r"""Test whether the given object or class is persistent, i.e.,
    whether it will save its source code when pickled.
    """
    try:
        if obj in _decorators:
            return True
    except TypeError:
        pass
    return type(obj) in _decorators # pylint: disable=unidiomatic-typecheck

def persistent_class(orig_class):
    r"""Class decorator that extends a given class to save its source code
    when pickled.

    Example:

        from torch_utils import persistence

        @persistence.persistent_class
        class MyNetwork(torch.nn.Module):
            def __init__(self, num_inputs, num_outputs):
                super().__init__()
                self.fc = MyLayer(num_inputs, num_outputs)
                ...

        @persistence.persistent_class
        class MyLayer(torch.nn.Module):
            ...

    When pickled, any instance of `MyNetwork` and `MyLayer` will save its
    source code alongside other internal state (e.g., parameters, buffers,
    and submodules). This way, any previously exported pickle will remain
    usable even if the class definitions have been modified or are no
    longer available.

    The decorator saves the source code of the entire Python module
    containing the decorated class. It does *not* save the source code of
    any imported modules. Thus, the imported modules must be available
    during unpickling, also including `torch_utils.persistence` itself.

    It is ok to call functions defined in the same module from the
    decorated class. However, if the decorated class depends on other
    classes defined in the same module, they must be decorated as well.
    This is illustrated in the above example in the case of `MyLayer`.

    It is also possible to employ the decorator just-in-time before
    calling the constructor. For example:

        cls = MyLayer
        if want_to_make_it_persistent:
            cls = persistence.persistent_class(cls)
        layer = cls(num_inputs, num_outputs)

    As an additional feature, the decorator also keeps track of the
    arguments that were used to construct each instance of the decorated
    class. The arguments can be queried via `obj.init_args` and
    `obj.init_kwargs`, and they are automatically pickled alongside other
    object state. This feature can be disabled on a per-instance basis
    by setting `self._record_init_args = False` in the constructor.

    A typical use case is to first unpickle a previous instance of a
    persistent class, and then upgrade it to use the latest version of
    the source code:

        with open('old_pickle.pkl', 'rb') as f:
            old_net = pickle.load(f)
        new_net = MyNetwork(*old_obj.init_args, **old_obj.init_kwargs)
        misc.copy_params_and_buffers(old_net, new_net, require_all=True)
    """
    assert isinstance(orig_class, type)
    if is_persistent(orig_class):
        return orig_class

    assert orig_class.__module__ in sys.modules
    orig_module = sys.modules[orig_class.__module__]
    orig_module_src = _module_to_src(orig_module)

    class Decorator(orig_class):
        _orig_module_src = orig_module_src
        _orig_class_name = orig_class.__name__

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            record_init_args = getattr(self, '_record_init_args', True)
            self._init_args = copy.deepcopy(args) if record_init_args else None
            self._init_kwargs = copy.deepcopy(kwargs) if record_init_args else None
            assert orig_class.__name__ in orig_module.__dict__
            _check_pickleable(self.__reduce__())

        @property
        def init_args(self):
            assert self._init_args is not None
            return copy.deepcopy(self._init_args)

        @property
        def init_kwargs(self):
            assert self._init_kwargs is not None
            return EasyDict(copy.deepcopy(self._init_kwargs))

        def __reduce__(self):
            fields = list(super().__reduce__())
            fields += [None] * max(3 - len(fields), 0)
            if fields[0] is not _reconstruct_persistent_obj:
                meta = dict(type='class', version=_version, module_src=self._orig_module_src, class_name=self._orig_class_name, state=fields[2])
                fields[0] = _reconstruct_persistent_obj # reconstruct func
                fields[1] = (meta,) # reconstruct args
                fields[2] = None # state dict
            return tuple(fields)

    Decorator.__name__ = orig_class.__name__
    Decorator.__module__ = orig_class.__module__
    _decorators.add(Decorator)
    return Decorator

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (th.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * th.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (th.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * th.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class AttentionOp(th.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = th.einsum('ncq,nck->nqk', q.to(th.float32), (k / np.sqrt(k.shape[1])).to(th.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = th._softmax_backward_data(grad_output=dw.to(th.float32), output=w.to(th.float32), dim=2, input_dtype=th.float32)
        dq = th.einsum('nck,nqk->ncq', k.to(th.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = th.einsum('ncq,nqk->nck', q.to(th.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


@persistent_class
class Linear(th.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = th.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = th.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

@persistent_class
class Conv2d(th.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = th.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = th.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = th.as_tensor(resample_filter, dtype=th.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = th.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = th.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = th.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = th.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = th.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = th.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = th.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

@persistent_class
class GroupNorm(th.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = th.nn.Parameter(th.ones(num_channels))
        self.bias = th.nn.Parameter(th.zeros(num_channels))

    def forward(self, x):
        x = th.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

@persistent_class
class PositionalEmbedding(th.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = th.arange(start=0, end=self.num_channels//2, dtype=th.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = th.cat([x.cos(), x.sin()], dim=1)
        return x

@persistent_class
class FourierEmbedding(th.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', th.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = th.cat([x.cos(), x.sin()], dim=1)
        return x

@persistent_class
class UNetBlock(th.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
        training_mode='', linear_probing=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.training_mode = training_mode
        self.linear_probing = linear_probing

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        if self.training_mode == 'ctm':
            self.affine_s = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

        if linear_probing:
            self.norm0_train = GroupNorm(num_channels=in_channels, eps=eps)
            self.conv0_train = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down,
                                      resample_filter=resample_filter, **init)
            self.norm1_train = GroupNorm(num_channels=out_channels, eps=eps)
            self.conv1_train = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

            self.skip_train = None
            if out_channels != in_channels or up or down:
                kernel = 1 if resample_proj or out_channels != in_channels else 0
                self.skip_train = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down,
                                   resample_filter=resample_filter, **init)

            if self.num_heads:
                self.norm2_train = GroupNorm(num_channels=out_channels, eps=eps)
                self.qkv_train = Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1,
                                  **(init_attn if init_attn is not None else init))
                self.proj_train = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb, emb_s=None, emb_t=None):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.training_mode == 'ctm':
            params_s = self.affine_s(emb_s).unsqueeze(2).unsqueeze(3).to(x.dtype)
            if not self.linear_probing:
                params = params + params_s
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(th.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(th.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = th.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale

        if self.linear_probing:
            y = self.conv0_train(silu(self.norm0_train(orig)))
            assert emb_t != None and self.training_mode == 'ctm'
            params_t = self.affine_s(emb_t).unsqueeze(2).unsqueeze(3).to(y.dtype)
            params = params_t - params_s
            params_ = params + params_s
            if self.adaptive_scale:
                scale, shift = params_.chunk(chunks=2, dim=1)
                y = silu(th.addcmul(shift, self.norm1_train(y), scale + 1))
            else:
                y = silu(self.norm1_train(y.add_(params_)))

            y = self.conv1_train(th.nn.functional.dropout(y, p=self.dropout, training=self.training))
            y = y.add_(self.skip_train(orig) if self.skip_train is not None else orig)
            y = y * self.skip_scale

            if self.num_heads:
                q, k, v = self.qkv_train(self.norm2_train(y)).reshape(y.shape[0] * self.num_heads, y.shape[1] // self.num_heads, 3,
                                                          -1).unbind(2)
                w = AttentionOp.apply(q, k)
                a = th.einsum('nqk,nck->ncq', w, v)
                y = self.proj_train(a.reshape(*y.shape)).add_(y)
                y = y * self.skip_scale
            y = y.mul(params)

            return x + y
        return x

@persistent_class
class SongUNet(th.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.13,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'fourier', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 2,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'residual',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,3,3,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        training_mode = '',
        linear_probing=False,
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']
        self.training_mode = training_mode
        self.linear_probing = linear_probing
        self.img_resolution = img_resolution

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        if self.training_mode.lower() == 'ctm':
            self.map_layer0_s = Linear(in_features=noise_channels, out_features=emb_channels, **init)
            self.map_layer1_s = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = th.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True,
                                                          training_mode=training_mode, linear_probing=linear_probing, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn,
                                                                training_mode=training_mode, linear_probing=linear_probing, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = th.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True,
                                                         training_mode=training_mode, linear_probing=linear_probing, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout,
                                                         training_mode=training_mode, linear_probing=linear_probing, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True,
                                                        training_mode=training_mode, linear_probing=linear_probing, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn,
                                                                training_mode=training_mode, linear_probing=linear_probing, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)
                if self.linear_probing:
                    self.dec[f'{res}x{res}_aux_norm_train'] = GroupNorm(num_channels=cout, eps=1e-6)
                    self.dec[f'{res}x{res}_aux_lin_train'] = Linear(in_features=emb_channels, out_features=out_channels*img_resolution*img_resolution, **init)
                    self.dec[f'{res}x{res}_aux_conv_train'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3,
                                                               **init_zero)

    def forward(self, x, noise_labels, noise_labels_s, class_labels):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (th.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))

        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))
        if noise_labels_s != None:
            emb_s = self.map_noise(noise_labels_s)
            emb_s = emb_s.reshape(emb_s.shape[0], 2, -1).flip(1).reshape(*emb_s.shape)  # swap sin/cos
            if self.map_label is not None:
                tmp = class_labels
                if self.training and self.label_dropout:
                    tmp = tmp * (th.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
                emb_s = emb_s + self.map_label(tmp * np.sqrt(self.map_label.in_features))
            emb_s = silu(self.map_layer0_s(emb_s))
            emb_s = silu(self.map_layer1_s(emb_s))
            if self.linear_probing:
                emb_t = self.map_noise(noise_labels)
                emb_t = emb_t.reshape(emb_t.shape[0], 2, -1).flip(1).reshape(*emb_t.shape)  # swap sin/cos
                emb_t = silu(self.map_layer0_s(emb_t))
                emb_t = silu(self.map_layer1_s(emb_t))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb, emb_s=None if noise_labels_s == None else emb_s,
                          emb_t=emb_t if self.linear_probing else None) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_lin' in name:
                emb_mult = (block(emb_t) - block(emb_s)).reshape(-1, 3, self.img_resolution, self.img_resolution)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp * emb_mult + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = th.cat([x, skips.pop()], dim=1)
                x = block(x, emb, emb_s=None if noise_labels_s == None else emb_s,
                          emb_t=emb_t if self.linear_probing else None)
        return aux

@persistent_class
class EDMPrecond_CTM(th.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'SongUNet',       # Class name of the underlying model.
        teacher = False,
        teacher_model_path = '',
        training_mode = '',
        arch='ncsn',
        linear_probing=False,
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.teacher = teacher
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.eye = th.eye(self.label_dim, device=dev)
        if teacher:
            # print("I am the teacher model -----------------")
            import pickle
            # print(f'Loading network from "{teacher_model_path}"...')
            with open(teacher_model_path, 'rb') as f:
                self.model = pickle.load(f)['ema']
        else:
            # print("I am not teacher *********************")
            if arch in ['ddpmpp', 'ncsnpp']:
                resample_filter = [1,1] if arch == 'ddpmpp' else [1,3,3,1]
                channel_mult_noise = 1 if arch == 'ddpmpp' else 2
                encoder_type = 'standard' if arch == 'ddpmpp' else 'residual'
                embedding_type = 'positional' if arch == 'ddpmpp' else 'fourier'
                self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels,
                                                   out_channels=img_channels, label_dim=label_dim,
                                                   training_mode=training_mode, resample_filter=resample_filter,
                                                   channel_mult_noise=channel_mult_noise, encoder_type=encoder_type,
                                                   embedding_type=embedding_type, linear_probing=linear_probing,
                                                   **model_kwargs)
            else:
                self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels,
                                                   out_channels=img_channels, label_dim=label_dim,
                                                   training_mode=training_mode, linear_probing=linear_probing,
                                                   **model_kwargs)

    def get_c_in(self, sigma):
        return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

    def unrescaling_t(self, rescaled_t):
        return th.exp(rescaled_t / 250.) - 1e-44

    def forward(self, rescaled_x, rescaled_t, s=None, teacher=False, **model_kwargs):
        class_labels = None if self.label_dim == 0 else th.zeros([1, self.label_dim], device=rescaled_x.device) \
            if model_kwargs == {} else self.eye[model_kwargs['y']].reshape(-1, self.label_dim)
        dtype = th.float16 if self.use_fp16 and rescaled_x.device.type == 'cuda' else th.float32
        if self.teacher:
            #with torch.no_grad():
            sigma = self.unrescaling_t(rescaled_t)
            c_in = append_dims(self.get_c_in(sigma), rescaled_x.ndim)
            x = rescaled_x / c_in
            D_x = self.model(x.to(dtype), sigma.flatten(), class_labels=class_labels)
            c_skip = append_dims(self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2), rescaled_x.ndim)
            c_out = append_dims(sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt(), rescaled_x.ndim)
            F_x = (D_x - c_skip * x) / c_out
        else:
            t = self.unrescaling_t(rescaled_t)
            t = t.log() / 4
            if s != None:
                s = self.unrescaling_t(s)
                s = s.log() / 4
            F_x = self.model(rescaled_x.to(dtype), t.flatten(), None if s == None else s.flatten(), class_labels=class_labels)
        #assert F_x.dtype == dtype
        return F_x

    def round_sigma(self, sigma):
        return th.as_tensor(sigma)

    def convert_to_fp16(self):
        pass

    def convert_to_fp32(self):
        pass


def DiffAugment(x, brightness, saturation, contrast, translation_x, translation_y,
                offset_x, offset_y, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        x = rand_brightness(x, brightness)
        x = rand_saturation(x, saturation)
        x = rand_contrast(x, contrast)
        x = rand_translation(x, translation_x, translation_y)
        x = rand_cutout(x, offset_x, offset_y)
        #for p in policy.split(','):
        #    for f in AUGMENT_FNS[p]:
        #        x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x

def rand_brightness(x, brightness):
    x = x + brightness
    return x

def rand_saturation(x, saturation):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * saturation + x_mean
    return x

def rand_contrast(x, contrast):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * contrast + x_mean
    return x

def rand_translation(x, translation_x, translation_y):
    grid_batch, grid_x, grid_y = th.meshgrid(
        th.arange(x.size(0), dtype=th.long, device=x.device),
        th.arange(x.size(2), dtype=th.long, device=x.device),
        th.arange(x.size(3), dtype=th.long, device=x.device),
    )
    grid_x = th.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = th.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

def rand_cutout(x, offset_x, offset_y, ratio=0.2):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    grid_batch, grid_x, grid_y = th.meshgrid(
        th.arange(x.size(0), dtype=th.long, device=x.device),
        th.arange(cutout_size[0], dtype=th.long, device=x.device),
        th.arange(cutout_size[1], dtype=th.long, device=x.device),
    )
    grid_x = th.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = th.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = th.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x

def get_weightings(weight_schedule, snrs, sigma_data, t, s, schedule_multiplier=None,):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    elif weight_schedule == "uniform_g":
        return 1./(1. - s / t) ** schedule_multiplier
    elif weight_schedule == "karras_weight":
        sigma = snrs ** -0.5
        weightings = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    elif weight_schedule == "sq-t-inverse":
        weightings = 1. / snrs ** 0.25
    else:
        raise NotImplementedError()
    return weightings

def gaussian_blur(args, img, step):
    blur_sigma = max(1 - step / (args.blur_fade_itr),
                     0) * args.blur_init_sigma if args.blur_fade_itr > 1 else 0
    blur_size = np.floor(blur_sigma * 3) * 2 - 1
    if blur_size > 1:
        img = torchvision.transforms.GaussianBlur(blur_size)(img)
    return img

def save(x, save_dir, name, npz=False):
    nrow = int(np.sqrt(x.shape[0]))
    image_grid = make_grid((x + 1.) / 2., nrow, padding=2)
    with bf.BlobFile(os.path.join(save_dir, f"{name}.png"), "wb") as fout:
        save_image(image_grid, fout)
    if npz:
        sample = ((x + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.cpu().detach()
        os.makedirs(os.path.join(save_dir, 'targets'), exist_ok=True)
        np.savez(os.path.join(save_dir, f"targets/{name}.npz"), sample.numpy())

def get_feature(args, input, feat, brightness, saturation, contrast, translation_x, translation_y,
                               offset_x, offset_y, name, step):
    # augment input
    input_aug_ = input
    if args.data_augment and brightness.shape[0] > 0:
        input_aug_ = DiffAugment(input[:brightness.shape[0]], brightness, saturation, contrast, translation_x, translation_y,
                                   offset_x, offset_y, policy='color,translation,cutout')
        input_aug_ = th.cat((input_aug_, input[brightness.shape[0]:]))
    # transform to [0,1]
    input_aug = input_aug_.add(1).div(2)
    # apply F-specific normalization
    input_n = Normalize(feat.normstats['mean'], feat.normstats['std'])(input_aug)
    # upsample if smaller, downsample if larger + VIT
    if input.shape[-2] < 256:
        input_n = F.interpolate(input_n, 224, mode='bilinear', align_corners=False)
        if args.save_png and step % args.save_period in [0, 1] and step >= 0:
            input_aug_ = F.interpolate(input_aug_, 224, mode='bilinear', align_corners=False)
            save(input_aug_, get_dir(), f'{name}_{step}_augment')
    # forward pass
    input_features = feat(input_n)
    return input_features

def get_xl_feature(args, estimate, target=None, feature_extractor=None, discriminator=None, step=-1, **model_kwargs):
    logits_fake, logits_real = [], []
    estimate_features, target_features = [], []
    for bb_name, feat in feature_extractor.items():
        # apply augmentation (x in [-1, 1])
        brightness = (th.rand(int(estimate.size(0) * args.prob_aug), 1, 1, 1, dtype=estimate.dtype,
                              device=estimate.device) - 0.5)
        # brightness = 0.
        saturation = (th.rand(int(estimate.size(0) * args.prob_aug), 1, 1, 1, dtype=estimate.dtype,
                              device=estimate.device) * 2)
        # saturation = 0.
        contrast = (th.rand(int(estimate.size(0) * args.prob_aug), 1, 1, 1, dtype=estimate.dtype,
                            device=estimate.device) + 0.5)
        # contrast = 0.
        shift_x, shift_y = int(estimate.size(2) * args.shift_ratio + 0.5), int(
            estimate.size(3) * args.shift_ratio + 0.5)
        translation_x = th.randint(-shift_x, shift_x + 1, size=[int(estimate.size(0) * args.prob_aug), 1, 1],
                                   device=estimate.device)
        translation_y = th.randint(-shift_y, shift_y + 1, size=[int(estimate.size(0) * args.prob_aug), 1, 1],
                                   device=estimate.device)
        cutout_size = int(estimate.size(2) * args.cutout_ratio + 0.5), int(
            estimate.size(3) * args.cutout_ratio + 0.5)
        offset_x = th.randint(0, estimate.size(2) + (1 - cutout_size[0] % 2),
                              size=[int(estimate.size(0) * args.prob_aug), 1, 1], device=estimate.device)
        offset_y = th.randint(0, estimate.size(3) + (1 - cutout_size[1] % 2),
                              size=[int(estimate.size(0) * args.prob_aug), 1, 1], device=estimate.device)

        estimate_feature = get_feature(args, estimate, feat, brightness, saturation, contrast,
                                            translation_x, translation_y, offset_x, offset_y, 'estimate',
                                            step)
        estimate_features.append(estimate_feature)
        if discriminator is not None:
            # print("printing discriminator:: ", discriminator)
            logits_fake += discriminator[bb_name](estimate_feature, model_kwargs)
            if args.gan_low_res_train:
                estimate_low_res = F.interpolate(estimate, 16, mode='bilinear', align_corners=False)
                if args.large_log:
                    print("estimate, estimate_low_res: ", estimate.shape, estimate_low_res.shape)
                estimate_feature_low_res = get_feature(args, estimate_low_res, feat, brightness, saturation, contrast,
                                               translation_x, translation_y, offset_x, offset_y, 'estimate_low_res',
                                               step)
                logits_fake += discriminator[bb_name + '_low'](estimate_feature_low_res, model_kwargs)

        if target != None:
            if args.gan_real_free and args.gan_different_augment:
                # apply augmentation (x in [-1, 1])
                brightness = (th.rand(int(target.size(0) * args.prob_aug), 1, 1, 1, dtype=target.dtype,
                                         device=target.device) - 0.5)
                # brightness = 0.
                saturation = (th.rand(int(target.size(0) * args.prob_aug), 1, 1, 1, dtype=target.dtype,
                                         device=target.device) * 2)
                # saturation = 0.
                contrast = (th.rand(int(target.size(0) * args.prob_aug), 1, 1, 1, dtype=target.dtype,
                                       device=target.device) + 0.5)
                # contrast = 0.
                shift_x, shift_y = int(target.size(2) * args.shift_ratio + 0.5), int(
                    target.size(3) * args.shift_ratio + 0.5)
                translation_x = th.randint(-shift_x, shift_x + 1, size=[int(target.size(0) * args.prob_aug), 1, 1],
                                              device=target.device)
                translation_y = th.randint(-shift_y, shift_y + 1, size=[int(target.size(0) * args.prob_aug), 1, 1],
                                              device=target.device)
                cutout_size = int(target.size(2) * args.cutout_ratio + 0.5), int(
                    target.size(3) * args.cutout_ratio + 0.5)
                offset_x = th.randint(0, target.size(2) + (1 - cutout_size[0] % 2),
                                         size=[int(target.size(0) * args.prob_aug), 1, 1], device=target.device)
                offset_y = th.randint(0, target.size(3) + (1 - cutout_size[1] % 2),
                                         size=[int(target.size(0) * args.prob_aug), 1, 1], device=target.device)
            with th.no_grad():
                target_feature = get_feature(args, target, feat, brightness, saturation, contrast,
                                                  translation_x, translation_y, offset_x, offset_y, 'target',
                                                  step)
                target_features.append(target_feature)
            if discriminator is not None:
                logits_real += discriminator[bb_name](target_feature, model_kwargs)
                if args.gan_low_res_train:
                    target_low_res = F.interpolate(target, 16, mode='bilinear', align_corners=False)
                    target_feature_low_res = get_feature(args, target_low_res, feat, brightness, saturation, contrast,
                                                           translation_x, translation_y, offset_x, offset_y,
                                                           'target_low_res', step)
                    logits_real += discriminator[bb_name + '_low'](target_feature_low_res, model_kwargs)
    if discriminator is not None:
        if target == None:
            return logits_fake
        else:
            return logits_fake, logits_real
    else:
        if target == None:
            return estimate_features
        else:
            return estimate_features, target_features

class KarrasDenoiser:
    def __init__(
        self,
        args,
        schedule_sampler,
        diffusion_schedule_sampler,
        feature_extractor=None,
        discriminator_feature_extractor=None,
    ):
        self.args = args
        self.schedule_sampler = schedule_sampler
        self.diffusion_schedule_sampler = diffusion_schedule_sampler
        self.feature_extractor = feature_extractor
        self.discriminator_feature_extractor = discriminator_feature_extractor
        self.num_timesteps = args.start_scales
        self.dist = nn.MSELoss(reduction='none')

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_c_in(self, sigma):
        return 1 / (sigma**2 + self.args.sigma_data**2) ** 0.5

    def get_inner_scalings(self, t, inner_parametrization='no'):
        if inner_parametrization == 'edm':
            c_skip, c_out = self.get_edm_scalings(t)
        elif inner_parametrization == 'cm':
            c_skip, c_out = self.get_cm_scalings(t)
        elif inner_parametrization == 'no':
            c_skip, c_out = th.zeros_like(t), th.ones_like(t)
        return c_skip, c_out

    def get_outer_scalings(self, t, s=None, outer_parametrization='euler'):
        if outer_parametrization == 'euler':
            c_skip = s / t
        elif outer_parametrization == 'variance':
            c_skip = (((s - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2) / (
                        (t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2)).sqrt()
        elif outer_parametrization == 'euler_variance_mixed':
            c_skip = s / (t + 1.) + \
                     (((s - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2) /
                      ((t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2)).sqrt() / (t + 1.)
        c_out = (1. - s / t)
        return c_skip, c_out

    def get_edm_scalings(self, sigma):
        c_skip = self.args.sigma_data**2 / (sigma**2 + self.args.sigma_data**2)
        c_out = sigma * self.args.sigma_data / (sigma**2 + self.args.sigma_data**2) ** 0.5
        return c_skip, c_out

    def get_cm_scalings(self, sigma):
        c_skip = self.args.sigma_data**2 / (
            (sigma - self.args.sigma_min) ** 2 + self.args.sigma_data**2
        )
        c_out = (
            (sigma - self.args.sigma_min)
            * self.args.sigma_data
            / (sigma**2 + self.args.sigma_data**2) ** 0.5
        )
        return c_skip, c_out

    def calculate_adaptive_weight(self, loss1, loss2, last_layer=None):
        loss1_grad = th.autograd.grad(loss1, last_layer, retain_graph=True)[0]
        loss2_grad = th.autograd.grad(loss2, last_layer, retain_graph=True)[0]
        d_weight = th.norm(loss1_grad) / (th.norm(loss2_grad) + 1e-4)
        d_weight = th.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight

    def rescaling_t(self, t):
        rescaled_t = 1000 * 0.25 * th.log(t + 1e-44)
        return rescaled_t

    def get_t(self, ind):
        if self.args.time_continuous:
            t = self.args.sigma_max ** (1 / self.args.rho) + ind * (
                    self.args.sigma_min ** (1 / self.args.rho) - self.args.sigma_max ** (1 / self.args.rho)
            )
            t = t ** self.args.rho
        else:
            t = self.args.sigma_max ** (1 / self.args.rho) + ind / (self.args.start_scales - 1) * (
                    self.args.sigma_min ** (1 / self.args.rho) - self.args.sigma_max ** (1 / self.args.rho)
            )
            t = t ** self.args.rho
        return t

    def get_num_heun_step(self, start_scales=-1, num_heun_step=-1, num_heun_step_random=None, heun_step_strategy='', time_continuous=None):
        if start_scales == -1:
            start_scales = self.args.start_scales
        if num_heun_step == -1:
            num_heun_step = self.args.hun_heun_step
        if num_heun_step_random == None:
            num_heun_step_random = self.args.num_heun_step_random
        if heun_step_strategy == '':
            heun_step_strategy = self.args.heun_step_strategy
        if time_continuous == None:
            time_continuous = self.args.time_continuous
        if num_heun_step_random:
            if time_continuous:
                num_heun_step = np.random.rand() * num_heun_step / start_scales
            else:
                if heun_step_strategy == 'uniform':
                    num_heun_step = np.random.randint(1,1+num_heun_step)
                elif heun_step_strategy == 'weighted':
                    p = np.array([i ** self.args.heun_step_multiplier for i in range(1,1+num_heun_step)])
                    p = p / sum(p)
                    num_heun_step = np.random.choice([i+1 for i in range(len(p))], size=1, p=p)[0]
        else:
            if time_continuous:
                num_heun_step = num_heun_step / start_scales
            else:
                num_heun_step = num_heun_step
        return num_heun_step

    def get_gan_time(self, x_start, noise, x_t, t, t_dt, s, indices, num_heun_step, gan_num_heun_step):
        if gan_num_heun_step != -1:
            if self.args.gan_specific_batch:
                bs = self.args.gan_micro_batch
                noise = th.randn([self.args.gan_micro_batch]+list(x_start.shape[1:]), device=x_start.device)
                assert self.args.gan_micro_batch % x_start.shape[0] == 0
                x_start = x_start.repeat(self.args.gan_micro_batch // x_start.shape[0], 1, 1, 1)
            else:
                bs = x_start.shape[0]
                noise = noise
            indices, _ = self.schedule_sampler.sample_t(bs, x_start.device, gan_num_heun_step,
                                                            self.args.time_continuous)
            new_indices = self.schedule_sampler.sample_s(self.args, bs, x_start.device,
                                                             indices,
                                                             gan_num_heun_step, self.args.time_continuous,
                                                             N=self.args.start_scales)
            t = self.get_t(indices)
            x_t = x_start + noise * append_dims(t, x_start.ndim)
            t_dt = self.get_t(indices + gan_num_heun_step)
            s = self.get_t(new_indices)
            num_heun_step = gan_num_heun_step
        return x_t, t, t_dt, s, indices, num_heun_step

    @th.no_grad()
    def heun_solver(self, target_model, x, ind, dims, t, t_dt, ctm=True, num_step=1, **model_kwargs):
        with th.no_grad():
            if self.args.self_learn:
                if self.args.self_learn_iterative:
                    for k in range(num_step):
                        t = self.get_t(ind + k)
                        t2 = self.get_t(ind + k + 1)
                        _, x = self.get_denoised_and_G(target_model, x, t, s=t2, ctm=ctm, **model_kwargs)
                else:
                    _, x = self.get_denoised_and_G(target_model, x, t, s=t_dt, ctm=ctm, **model_kwargs)
            else:
                for k in range(num_step):
                    t = self.get_t(ind + k)
                    denoised, _ = self.get_denoised_and_G(self.teacher_model, x, t, None, ctm=False, teacher=True, **model_kwargs)
                    d = (x - denoised) / append_dims(t, dims)
                    t2 = self.get_t(ind + k + 1)
                    x_phi_ODE_1st = x + d * append_dims(t2 - t, dims)
                    denoised2, _ = self.get_denoised_and_G(self.teacher_model, x_phi_ODE_1st, t2, None, ctm=False, teacher=True, **model_kwargs)
                    next_d = (x_phi_ODE_1st - denoised2) / append_dims(t2, dims)
                    x_phi_ODE_2nd = x + (d + next_d) * append_dims((t2 - t) / 2, dims)
                    x = x_phi_ODE_2nd
            return x

    def get_gan_fake(self, estimate, x_t, t, t_dt, s, model, target_model, ctm, step, **model_kwargs):
        if self.args.gan_fake_outer_type == 'no':
            _, fake = self.get_denoised_and_G(model, x_t, t, s=th.ones_like(s) * self.args.sigma_min if s != None else None, ctm=ctm, **model_kwargs)
        else:
            assert self.args.gan_fake_outer_type in ['model', 'target_model_sg']
            assert self.args.gan_fake_inner_type in ['model', 'model_sg', 'target_model_sg']
            if (self.args.gan_fake_outer_type == self.args.ctm_estimate_outer_type) \
                and (self.args.gan_fake_inner_type == self.args.ctm_estimate_inner_type):
                    fake = estimate
            else:
                fake = self.get_ctm_estimate(x_t, t, t_dt, s, model, target_model, ctm,
                                             outer_type=self.args.gan_fake_outer_type,
                                             inner_type=self.args.gan_fake_inner_type,
                                             target_matching=self.args.gan_target_matching, **model_kwargs)
        if self.args.gaussian_filter:
            fake = gaussian_blur(self.args, fake, step)
        return fake

    @th.no_grad()
    def get_gan_real(self, x_start, x_t, t, t_dt, s, indices, dims, num_heun_step, model, target_model, ctm, step, **model_kwargs):
        with th.no_grad():
            if self.args.gan_real_free:
                real = x_start
            else:
                x_t_dt = self.heun_solver(target_model, x_t, indices, dims, t, t_dt, ctm=ctm, num_step=num_heun_step,
                                          **model_kwargs).detach()
                #if self.args.gan_real_inner_type == 'no':
                #    _, real = self.get_denoised_and_G(target_model, x_t_dt, t_dt, s=th.ones_like(s) * self.args.sigma_min,
                #                                      ctm=ctm, **model_kwargs)
                #else:
                assert self.args.gan_real_inner_type in ['model_sg', 'target_model_sg', 'no']
                real = self.get_ctm_target(x_t_dt, t_dt, s, model, target_model, ctm,
                                           self.args.gan_real_inner_type, **model_kwargs)
            if self.args.gaussian_filter:
                real = gaussian_blur(self.args, real, step).detach()
        return real.detach()

    def get_ctm_estimate(self, x_t, t, t_dt, s, model, target_model, ctm, outer_type, inner_type, target_matching, **model_kwargs):
        if self.args.large_log:
            print("CTM estimate inner type, outer type, ctm: ", inner_type, outer_type, ctm)
        if target_matching:
            s = t_dt
        if inner_type == 'model':
            _, estimate = self.get_denoised_and_G(model, x_t, t, s=s, ctm=ctm, **model_kwargs)
        elif inner_type == 'model_sg':
            with th.no_grad():
                _, estimate = self.get_denoised_and_G(model, x_t, t, s=s, ctm=ctm, **model_kwargs)
        elif inner_type == 'target_model_sg':
            with th.no_grad():
                _, estimate = self.get_denoised_and_G(target_model, x_t, t, s=s, ctm=ctm, **model_kwargs)
        else:
            raise NotImplementedError
        if self.args.training_mode == 'ctm':
            if outer_type == 'model':
                _, estimate = self.get_denoised_and_G(model, estimate, s, s=th.ones_like(s) * self.args.sigma_min,
                                             ctm=ctm, **model_kwargs)
            elif outer_type == 'target_model_sg':
                _, estimate = self.get_denoised_and_G(target_model, estimate, s, s=th.ones_like(s) * self.args.sigma_min,
                                             ctm=ctm, **model_kwargs)
            else:
                raise NotImplementedError
        return estimate

    @th.no_grad()
    def get_ctm_target(self, x_t_dt, t_dt, s, model, target_model, ctm, inner_type, **model_kwargs):
        if self.args.large_log:
            print("CTM target inner type, ctm: ", inner_type, ctm)
        with th.no_grad():
            if inner_type == 'model_sg':
                _, target = self.get_denoised_and_G(model, x_t_dt, t_dt, s=s, ctm=ctm, **model_kwargs)
            elif inner_type == 'target_model_sg':
                _, target = self.get_denoised_and_G(target_model, x_t_dt, t_dt, s=s, ctm=ctm, **model_kwargs)
            elif inner_type == 'no':
                target = x_t_dt
                s = t_dt
            else:
                raise NotImplementedError
            if self.args.training_mode == 'ctm':
                _, target = self.get_denoised_and_G(target_model, target, s, s=th.ones_like(s) * self.args.sigma_min, ctm=ctm, **model_kwargs)
            return target.detach()

    def get_denoised(self, g_theta, x_t ,t):
        if self.args.outer_parametrization.lower() == 'euler':
            denoised = g_theta
        elif self.args.outer_parametrization.lower() == 'variance':
            denoised = g_theta + append_dims((self.args.sigma_min ** 2 + self.args.sigma_data ** 2
                                              - self.args.sigma_min * t) / \
                                             ((t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2),
                                             x_t.ndim) * x_t
        elif self.args.outer_parametrization.lower() == 'euler_variance_mixed':
            denoised = g_theta + x_t - append_dims(t / (t + 1.) * (1. + (t - self.args.sigma_min) /
                                                                   ((
                                                                                t - self.args.sigma_min) ** 2 + self.args.sigma_data ** 2)),
                                                   x_t.ndim) * x_t
        else:
            raise NotImplementedError
        return denoised

    def get_denoised_and_G(self, model, x_t, t, s=None, ctm=False, teacher=False, **model_kwargs):
        rescaled_t = self.rescaling_t(t)
        if s != None:
            rescaled_s = self.rescaling_t(s)
        else:
            rescaled_s = None
        c_in = append_dims(self.get_c_in(t), x_t.ndim)
        # print('printing the model ----------->>>', model)
        # print('c_in --------------->>>>', c_in)
        # print('c_in --------------->>>>', x_t)
        model_output = model(c_in * x_t, rescaled_t, s=rescaled_s, teacher=teacher, **model_kwargs)

        if ctm:
            if self.args.target_subtract:
                with th.no_grad():
                    teacher_denoised = self.teacher_model(c_in * x_t, rescaled_t, s=None, teacher=True, **model_kwargs)
                if self.args.rescaling:
                    model_output = model_output * append_dims((t ** 2 - s ** 2) ** 0.5 / t, x_t.ndim)
                model_output = model_output + teacher_denoised
            c_skip, c_out = [
                append_dims(x, x_t.ndim)
                for x in self.get_inner_scalings(t, self.args.inner_parametrization)
            ]
            g_theta = c_out * model_output + c_skip * x_t
            #z = th.randn_like(x_t)
            #x_t_ = x_t + 0.001 * z
            #model_output_ = model(c_in * x_t_, rescaled_t, s=rescaled_s, teacher=teacher, **model_kwargs)
            #g_theta_ = c_out * model_output_ + c_skip * x_t
            #np.savez(bf.join(logger.get_dir(), f"g/{np.random.randint(10000000)}"), \
            #         x_t=x_t.cpu().detach().numpy(), t=t.reshape(-1).cpu().detach().numpy(),
            #         s=s.reshape(-1).cpu().detach().numpy(), g=g_theta.cpu().detach().numpy(),
            #         g_=g_theta_.cpu().detach().numpy(), z=z.cpu().detach().numpy())
            denoised = self.get_denoised(g_theta, x_t, t)
            if s == None:
                return denoised, denoised
            c_skip, c_out = [
                append_dims(x, x_t.ndim)
                for x in self.get_outer_scalings(t, s, self.args.outer_parametrization)
            ]
            G_theta = c_out * g_theta + c_skip * x_t
        else:
            if teacher:
                c_skip, c_out = [
                    append_dims(x, x_t.ndim) for x in self.get_edm_scalings(t)
                ]
            else:
                c_skip, c_out = [
                    append_dims(x, x_t.ndim)
                    for x in self.get_cm_scalings(t)
                ]
            denoised = c_out * model_output + c_skip * x_t
            G_theta = denoised
        return denoised, G_theta

    def get_CTM_loss(self, estimate, target, weights, step):
        if self.args.loss_norm == 'lpips':
            if estimate.shape[-2] < 256:
                estimate = F.interpolate(estimate, size=224, mode="bilinear")
                target = F.interpolate(
                    target, size=224, mode="bilinear"
                )
            consistency_loss = (self.feature_extractor(
                (estimate + 1) / 2.0,
                (target + 1) / 2.0, ) * weights)
        elif self.args.loss_norm == "cnn_vit":
            distances, estimate_features, target_features = [], [], []
            estimate_features, target_features = get_xl_feature(self.args, estimate, target,
                                                                     feature_extractor=self.feature_extractor, step=step)
            cnt = 0
            for _, _ in self.feature_extractor.items():
                for fe in list(estimate_features[cnt].keys()):
                    norm_factor = th.sqrt(th.sum(estimate_features[cnt][fe] ** 2, dim=1, keepdim=True))
                    est_feat = estimate_features[cnt][fe] / (norm_factor + 1e-10)
                    norm_factor = th.sqrt(th.sum(target_features[cnt][fe] ** 2, dim=1, keepdim=True))
                    tar_feat = target_features[cnt][fe] / (norm_factor + 1e-10)
                    distances.append(self.dist(est_feat, tar_feat))
                cnt += 1
            consistency_loss = th.cat([d.mean(dim=[2, 3]) for d in distances], dim=1).sum(dim=1)
        else:
            raise NotImplementedError
        return consistency_loss

    def get_DSM_loss(self, model, x_start, model_kwargs, consistency_loss,
                           step, init_step):
        sigmas, denoising_weights = self.diffusion_schedule_sampler.sample(x_start.shape[0], dev)
        noise = th.randn_like(x_start)
        dims = x_start.ndim
        x_t = x_start + noise * append_dims(sigmas, dims)
        denoised, _ = self.get_denoised_and_G(model, x_t, sigmas, s=sigmas if self.args.training_mode == 'ctm' else None, ctm=True, teacher=True, **model_kwargs)
        snrs = self.get_snr(sigmas)
        denoising_weights = append_dims(get_weightings(self.args.diffusion_weight_schedule, snrs, self.args.sigma_data, None, None), dims)
        denoising_loss = mean_flat(denoising_weights * (denoised - x_start) ** 2)
        if self.args.apply_adaptive_weight:
            if self.args.data_name.lower() == 'cifar10':
                # print(model)
                balance_weight = self.calculate_adaptive_weight(consistency_loss.mean(), denoising_loss.mean(),
                                                                last_layer=model.model.dec[
                                                                    '32x32_aux_conv'].weight)
            else:
                balance_weight = self.calculate_adaptive_weight(consistency_loss.mean(), denoising_loss.mean(),
                                                                last_layer=
                                                                model.module.output_blocks[15][0].out_layers[
                                                                    3].weight)
        else:
            balance_weight = 1.
        if self.args.large_log:
            log("denoising weight: ", balance_weight)
        balance_weight = self.adopt_weight(balance_weight, step, threshold=init_step, value=1.)
        denoising_loss = denoising_loss * balance_weight
        return denoising_loss

    def get_GAN_loss(self, model, real=None, fake=None, consistency_loss=None,
                               learn_generator=True, discriminator=None, step=0, init_step=0, **model_kwargs):

        if learn_generator:
            logits_fake = get_xl_feature(self.args, fake, feature_extractor=self.discriminator_feature_extractor,
                                                  discriminator=discriminator, **model_kwargs)
            g_loss = sum([(-l).mean() for l in logits_fake]) / len(logits_fake)
            if self.args.large_log:
                log("g_loss: ", g_loss.mean().item())
            if self.args.d_apply_adaptive_weight:
                CTM_loss = consistency_loss.mean()
                if self.args.data_name.lower() == 'cifar10':
                    d_weight = self.calculate_adaptive_weight(CTM_loss.mean(),
                                                              g_loss.mean(),
                                                              last_layer=model.model.dec[
                                                                  '32x32_aux_conv'].weight)
                else:
                    d_weight = self.calculate_adaptive_weight(CTM_loss.mean(),
                                                              g_loss.mean(),
                                                              last_layer=
                                                              model.module.output_blocks[15][0].out_layers[3].weight)
                d_weight = th.clip(d_weight, 0.01, 10.)
            else:
                d_weight = 1.
            discriminator_loss = self.adopt_weight(d_weight, step,
                                                   threshold=init_step + self.args.discriminator_start_itr) * g_loss
        else:
            logits_fake, logits_real = get_xl_feature(self.args, fake.detach(), target=real.detach(),
                                                      feature_extractor=self.discriminator_feature_extractor,
                                                      discriminator=discriminator, step=step, **model_kwargs)
            loss_Dgen = sum([(F.relu(th.ones_like(l) + l)).mean() for l in logits_fake]) / len(logits_fake)
            loss_Dreal = sum([(F.relu(th.ones_like(l) - l)).mean() for l in logits_real]) / len(logits_real)
            discriminator_loss = loss_Dreal + loss_Dgen
            if self.args.large_log:
                print("logits_real: ", sum([l.mean() for l in logits_real]).item() / len(logits_real), len(logits_real))
                print("logits_fake: ", sum([l.mean() for l in logits_fake]).item() / len(logits_fake), len(logits_fake))
        return discriminator_loss

    def ctm_losses(
        self,
        step,
        model,
        x_start,
        model_kwargs=None,
        target_model=None,
        noise=None,
        discriminator=None,
        init_step=0,
        ctm=True,
        num_heun_step=-1,
        gan_num_heun_step=-1,
        diffusion_training_=False,
        gan_training_=False,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        dims = x_start.ndim
        s = None
        terms = {}
        if num_heun_step == -1:
            num_heun_step = [self.get_num_heun_step(num_heun_step=self.args.num_heun_step)]
            dist.broadcast_object_list(num_heun_step, 0)
            num_heun_step = num_heun_step[0]
        if self.args.large_log:
            print("x batch size: ", x_start.shape)
            print("num heun step: ", num_heun_step)
            print("diffusion training: ", diffusion_training_)
            print("gan training: ", gan_training_)
        indices, _ = self.schedule_sampler.sample_t(x_start.shape[0], x_start.device, num_heun_step,
                                                    self.args.time_continuous)
        t = self.get_t(indices)
        t_dt = self.get_t(indices + num_heun_step)
        if ctm:
            new_indices = self.schedule_sampler.sample_s(self.args, x_start.shape[0], x_start.device, indices,
                                                         num_heun_step, self.args.time_continuous,
                                                         N=self.args.start_scales)
            s = self.get_t(new_indices)
        x_t = x_start + noise * append_dims(t, dims)
        dropout_state = th.get_rng_state()
        th.set_rng_state(dropout_state)
        if self.args.ctm_training:
            ctm_estimate = self.get_ctm_estimate(x_t, t, t_dt, s, model, target_model, ctm=ctm,
                                                 outer_type=self.args.ctm_estimate_outer_type,
                                                 inner_type=self.args.ctm_estimate_inner_type,
                                                 target_matching=self.args.ctm_target_matching,
                                                 **model_kwargs)
        if step % self.args.g_learning_period == 0 or not self.args.gan_training:
            if self.args.ctm_training:
                assert (discriminator == None) == (self.args.g_learning_period == 1)
                x_t_dt = self.heun_solver(target_model, x_t, indices, dims, t, t_dt, ctm=ctm, num_step=num_heun_step,
                                          **model_kwargs).detach()
                ctm_target = self.get_ctm_target(x_t_dt, t_dt, s, model, target_model, ctm=ctm,
                                                 inner_type=self.args.ctm_target_inner_type, **model_kwargs)

                if self.args.save_png and step % self.args.save_period == 0:
                    save(ctm_estimate, get_dir(), f'ctm_estimate_{step}')  # _{r}')
                    save(ctm_target, get_dir(), f'ctm_target_{step}')  # _{r}')
                    save(x_t, get_dir(), f'non_denoised_{step}')  # _{r}')
                    save(x_t_dt, get_dir(), f'denoised_{step}')  # _{r}')

                snrs = self.get_snr(t)
                weights = get_weightings(self.args.weight_schedule, snrs, self.args.sigma_data, t, s, self.args.weight_schedule_multiplier)

                terms["consistency_loss"] = self.get_CTM_loss(ctm_estimate, ctm_target, weights, step - init_step,)
                if self.args.large_log:
                    if s != None:
                        print(f"{step}-th step, t, t-dt, s, weight, loss: ", t[0].item(), t_dt[0].item(), s[0].item(), weights[0].item(), terms["consistency_loss"][0].item())
                    else:
                        print(f"{step}-th step, t, t-dt, weight, loss: ", t[0].item(), t_dt[0].item(), terms["consistency_loss"][0].item(), weights[0].item())
            if self.args.diffusion_training:
                if diffusion_training_:
                    terms['denoising_loss'] = self.get_DSM_loss(model, x_start, model_kwargs,
                                                                      terms["consistency_loss"] if self.args.ctm_training else None,
                                                                      step, init_step)

            if self.args.gan_training and step - init_step >= self.args.discriminator_start_itr:
                if gan_training_:
                    gan_x_t, gan_t, gan_t_dt, gan_s, _, _ = self.get_gan_time(x_start, noise, x_t, t, t_dt, s, indices,
                                                                              num_heun_step, gan_num_heun_step)
                    #if self.args.gan_specific_time:
                    #    print("gan_specific_time: ", gan_t.reshape(-1)[:3], gan_t_dt.reshape(-1)[:3], gan_s.reshape(-1)[:3])
                    gan_fake = self.get_gan_fake(ctm_estimate, gan_x_t, gan_t, gan_t_dt, gan_s, model, target_model, ctm,
                                                 step - init_step, **model_kwargs)
                    if self.args.gan_specific_batch:
                        if self.args.large_log:
                            print("gan_specific_micro_batch: ", gan_fake.shape, self.args.gan_micro_batch)
                    terms['d_loss'] = self.get_GAN_loss(model, fake=gan_fake,
                                                                  consistency_loss=terms["consistency_loss"],
                                                                  discriminator=discriminator,
                                                                  step=step, init_step=init_step)
        else:
            gan_x_t, gan_t, gan_t_dt, gan_s, gan_indices, gan_num_heun_step = \
                self.get_gan_time(x_start, noise, x_t, t, t_dt, s, indices, num_heun_step, gan_num_heun_step)
            gan_real = self.get_gan_real(x_start, gan_x_t, gan_t, gan_t_dt, gan_s, gan_indices, dims, gan_num_heun_step,
                                         model, target_model, ctm, step - init_step, **model_kwargs)
            gan_fake = self.get_gan_fake(ctm_estimate, gan_x_t, gan_t, gan_t_dt, gan_s, model, target_model, ctm,
                                         step - init_step, **model_kwargs)
            terms['d_loss'] = self.get_GAN_loss(model, fake=gan_fake, real=gan_real,
                                                learn_generator=False, discriminator=discriminator,
                                                step=step, init_step=init_step, **model_kwargs)
        return terms

# from cm.karras_diffusion import KarrasDenoiser


def create_model_and_diffusion(args, feature_extractor=None, discriminator_feature_extractor=None, teacher=False):
    schedule_sampler = create_named_schedule_sampler(args, args.schedule_sampler, args.start_scales)
    diffusion_schedule_sampler = create_named_schedule_sampler(args, args.diffusion_schedule_sampler, args.start_scales)
    if args.data_name.lower() == 'cifar10':
        # from cm.networks import EDMPrecond_CTM
        # print("printing the teacher model path ------------>>>>", args.teacher_model_path)
        model = EDMPrecond_CTM(img_resolution=args.image_size, img_channels=3,
                               label_dim=1000 if args.data_name.lower() == 'imagenet64' else 10 if args.class_cond else 0, use_fp16=args.use_fp16,
                               sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                               sigma_data=args.sigma_data, model_type='SongUNet' if args.data_name.lower() == 'cifar10' else 'DhariwalUNet',
                               teacher=teacher, teacher_model_path=args.teacher_model_path or args.model_path,
                               training_mode=args.training_mode, arch='ddpmpp' if args.data_name.lower() == 'cifar10' else 'adm',
                               linear_probing=args.linear_probing)
    # print("loaded the edm precond ctm model")
    # print("done in create model and diffusion")
    # exit()
    diffusion = KarrasDenoiser(
        args=args, schedule_sampler=schedule_sampler,
        diffusion_schedule_sampler=diffusion_schedule_sampler,
        feature_extractor=feature_extractor,
        discriminator_feature_extractor=discriminator_feature_extractor,
    )
    return model, diffusion


###### Training loop starts here

INITIAL_LOG_LOSS_SCALE = 20.0

def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar = []
    matrix = []
    for n, p in named_model_params:
        if p.ndim <= 1 and p.requires_grad:
            scalar.append((n, p))
        if p.ndim > 1 and p.requires_grad:
            matrix.append((n, p))
    scalar_vector_named_params = (
        scalar,
        (-1),
    )
    matrix_named_params = (
        matrix,
        (1, -1),
    )

    return [scalar_vector_named_params, matrix_named_params]

def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        #a = []
        #for _, param in param_group:
        #    if param.requires_grad == True:
        #        a.append([param.detach().float()])
        #master_param = nn.Parameter(
        #    _flatten_dense_tensors(a).view(shape)
        #)
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params

def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None

def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)

def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)

def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)

def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])

def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict

def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [
            (name, state_dict[name].to(dev)) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params

def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)

class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        #for name, param in self.model.named_parameters():
        #    print(name, param.requires_grad)

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor):
        if self.use_fp16:
            loss_scale = 2**self.lg_loss_scale
            #print("loss value: ", (loss*loss_scale).item())
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize(self, opt: th.optim.Optimizer):
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: th.optim.Optimizer):
        logkv_mean("lg_loss_scale", self.lg_loss_scale)
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2**self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False

        logkv_mean("grad_norm", grad_norm)
        logkv_mean("param_norm", param_norm)

        for p in self.master_params:
            p.grad.mul_(1.0 / (2**self.lg_loss_scale))

        opt.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer):
        # model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms()
        logkv_mean("grad_norm", grad_norm)
        logkv_mean("param_norm", param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


class DummyGenerator:
    def randn(self, *args, **kwargs):
        return th.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return th.randint(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        return th.randn_like(*args, **kwargs)

class DeterministicGenerator:
    """
    RNG to deterministically sample num_samples samples that does not depend on batch_size or mpi_machines
    Uses a single rng and samples num_samples sized randomness and subsamples the current indices
    """

    def __init__(self, num_samples, seed=0):

        self.num_samples = num_samples
        self.done_samples = 0
        self.seed = seed
        self.rng_cpu = th.Generator()
        if th.cuda.is_available():
            self.rng_cuda = th.Generator(dev)
        self.set_seed(seed)

    def get_global_size_and_indices(self, size):
        global_size = (self.num_samples, *size[1:])
        indices = th.arange(self.done_samples, self.done_samples + size[0])
        indices = th.clamp(indices, 0, self.num_samples - 1)
        return global_size, indices

    def get_generator(self, device):
        return self.rng_cpu if th.device(device).type == "cpu" else self.rng_cuda

    def randn(self, *size, dtype=th.float, device="cpu"):
        global_size, indices = self.get_global_size_and_indices(size)
        generator = self.get_generator(device)
        return th.randn(*global_size, generator=generator, dtype=dtype, device=device)[indices]

    def randint(self, low, high, size, dtype=th.long, device="cpu"):
        global_size, indices = self.get_global_size_and_indices(size)
        generator = self.get_generator(device)
        return th.randint(low, high, generator=generator, size=global_size, dtype=dtype, device=device)[indices]

    def randn_like(self, tensor):
        size, dtype, device = tensor.size(), tensor.dtype, tensor.device
        return self.randn(*size, dtype=dtype, device=device)

    def set_done_samples(self, done_samples):
        self.done_samples = done_samples
        self.set_seed(self.seed)

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.rng_cpu.manual_seed(seed)
        if th.cuda.is_available():
            self.rng_cuda.manual_seed(seed)


def get_generator(generator, num_samples=0, seed=0):
    if generator == "dummy":
        return DummyGenerator()
    elif generator == "determ":
        return DeterministicGenerator(num_samples, seed)
    else:
        print("check if the necessary module has been removed. check in cm.random_util.py")
        raise NotImplementedError

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

from torch.optim import RAdam

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def get_target_param_groups_and_shapes(named_model_params, source_named_model_params):
    named_model_params = list(named_model_params)
    source_named_model_params = {n: p for (n,p) in list(source_named_model_params)}
    scalar = []
    matrix = []
    for n, p in named_model_params:
        if p.ndim <= 1 and source_named_model_params[n].requires_grad:
            scalar.append((n, p))
        if p.ndim > 1 and source_named_model_params[n].requires_grad:
            matrix.append((n, p))
    scalar_vector_named_params = (
        scalar,
        (-1),
    )
    matrix_named_params = (
        matrix,
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]

def karras_sample(
    diffusion,
    model,
    shape,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
    x_T=None,
    ctm=False,
    teacher=False,
    clip_output=True,
    train=False,
    ind_1=0,
    ind_2=0,
    gamma=0.5,
):
    if generator is None:
        generator = get_generator("dummy")

    if sampler in ["progdist", 'euler', 'exact', 'cm_multistep', 'gamma_multistep']:
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    if x_T == None:
        x_T = generator.randn(*shape, device=device) * sigma_max

    sample_fn = {
        "heun": sample_heun,
        # "dpm": sample_dpm,
        # "ancestral": sample_euler_ancestral,
        # "onestep": sample_onestep,
        "exact": sample_exact,
        # "gamma": sample_gamma,
        # "gamma_multistep": sample_gamma_multistep,
        # "progdist": sample_progdist,
        # "euler": sample_euler,
        # "multistep": stochastic_iterative_sampler,
        # "cm_multistep": sample_multistep,
    }[sampler]
    # print(sampler)
    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler in ["multistep", "exact", "cm_multistep"]:
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=rho, steps=steps
        )
    elif sampler in ["gamma"]:
        sampler_args = dict(ind_1=ind_1, ind_2=ind_2)
    elif sampler in ["gamma_multistep"]:
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=rho, steps=steps, gamma=gamma,
        )
    else:
        sampler_args = {}
    if sampler in ['heun']:
        sampler_args['teacher'] = False if train else teacher
        sampler_args['ctm'] = ctm
    #print("clip_denoised, clip_output: ", clip_denoised, clip_output)
    def denoiser(x_t, t, s=th.ones(x_T.shape[0], device=device)):
        denoised, G_theta = diffusion.get_denoised_and_G(model, x_t, t, s, ctm, teacher, **model_kwargs)
        if sampler in ['exact', 'cm_multistep', 'onestep', 'gamma', 'gamma_multistep']:
            denoised = G_theta
        if clip_denoised:
            #print("clip denoised!!!")
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )
    if clip_output:
        #print("clip output")
        return x_0.clamp(-1, 1)
    return x_0

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    teacher=False,
    ctm=False,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        print("sigmas: ", sigmas[i], ctm, teacher)
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        if ctm:
            denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in)
        else:
            #if teacher:
            denoised = denoiser(x, sigma_hat * s_in, s=None)
            #else:
            #    denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in)
        #print("denoised: ", denoised[0][0][0][:3])
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
            #print("last")
        else:
            #print("no last")
            # Heun's method
            x_2 = x + d * dt
            if ctm:
                denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in)
            else:
                #if teacher:
                denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=None)
                #else:
                #    denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x

@th.no_grad()
def sample_exact(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)


    for i in indices[:-1]:
        sigma = sigmas[i]
        print(sigma, sigmas[i+1])
        if sigmas[i+1] != 0:
            denoised = denoiser(x, sigma * s_in, s=sigmas[i + 1] * s_in)
            x = denoised
        else:
            denoised = denoiser(x, sigma * s_in, s=sigma * s_in)
            d = to_d(x, sigma, denoised)
            dt = sigmas[i + 1] - sigma
            x = x + d * dt
        #else:
        #    denoised = denoiser(x, sigma * s_in)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        #x = denoised
    return x

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

def append_zero(x):
    return th.cat([x, x.new_zeros([1])])


class TrainLoop:
    def __init__(
        self,
        #*,
        model,
        discriminator,
        diffusion,
        data,
        batch_size,
        args=None,
    ):
        self.args = args
        self.model = model
        if self.args.sanity_check:
            for name, param in self.model.named_parameters():
                log("check and understand how consistency-type models override model parameters")
                log("model parameter before overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                break
        self.discriminator = discriminator
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = args.microbatch if args.microbatch > 0 else batch_size
        self.lr = args.lr
        self.ema_rate = (
            [args.ema_rate]
            if isinstance(args.ema_rate, float)
            else [float(x) for x in args.ema_rate.split(",")]
        )
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size #* dist.get_world_size()
        self.fids = []
        self.generator = get_generator('determ', self.args.eval_num_samples, self.args.eval_seed)
        self.x_T = self.generator.randn(*(self.args.sampling_batch, self.args.in_channels, self.args.image_size, self.args.image_size),
                                        device='cpu') * self.args.sigma_max #.to(dist_util.dev())
        if self.args.class_cond:
            self.classes = self.generator.randint(0, self.args.num_classes, (self.args.sampling_batch,), device='cpu')
            if self.args.data_name.lower() == 'cifar10':
                self.classes.sort()

        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()
        if self.args.sanity_check:
            for name, param in self.model.named_parameters():
                log("model parameter after overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                break
        if self.discriminator != None:
            if self.args.d_opt_load:
                if self.args.sanity_check:
                    for name, param in self.discriminator.named_parameters():
                        log("discriminator parameter before overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                        break
                self._load_and_sync_discriminator_parameters()
                if self.args.sanity_check:
                    for name, param in self.discriminator.named_parameters():
                        log("discriminator parameter after overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                        break
        print("loaded discriminator")
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
        )
        print("loaded mixed precision trainer")
        if self.args.sanity_check:
            log("mp trainer master parameter (should same to the model parameter if no linear_probing): ", self.mp_trainer.master_params[1].reshape(-1)[:3])

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.args.weight_decay
        )
        #if self.args.sanity_check:
        #    print("opt state dict before overriding: ", self.opt.state_dict())

        if self.discriminator != None:
            self.d_mp_trainer = MixedPrecisionTrainer(
                model=self.discriminator,
                use_fp16=args.use_d_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
            )
            self.d_opt = RAdam(
                self.d_mp_trainer.master_params, lr=args.d_lr, weight_decay=self.args.weight_decay, betas=(0.5, 0.9)
            )
        print('going to resume step')
        if self.resume_step:
            self._load_optimizer_state()
            if self.discriminator != None:
                try:
                    #if self.args.sanity_check:
                    #    print("discriminator opt state dict before overriding: ", self.d_opt.state_dict())
                    self._load_d_optimizer_state()
                    #if self.args.sanity_check:
                    #    print("discriminator opt state dict after overriding: ", self.d_opt.state_dict())
                except:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!! warning !!!!!!!!!!!!!!!!!!!!!!!!!!!! discriminator optimizer not loaded successfully")
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = self.model #DDP(
            #     self.model,
            #     device_ids=[dist_util.dev()],
            #     output_device=dist_util.dev(),
            #     broadcast_buffers=False,
            #     bucket_cap_mb=128,
            #     find_unused_parameters=False,
            # )
            self.ddp_discriminator = None
            if self.args.gan_training:
                self.ddp_discriminator = self.discriminator #DDP(
                #     self.discriminator,
                #     device_ids=[dist_util.dev()],
                #     output_device=dist_util.dev(),
                #     broadcast_buffers=False,
                #     bucket_cap_mb=128,
                #     find_unused_parameters=False,
                # )

        else:
            # if dist.get_world_size() > 1:
            #     logger.warn(
            #         "Distributed training requires CUDA. "
            #         "Gradients will not be synchronized properly!"
            #     )
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            log(f"loading pretrained model from checkpoint: {resume_checkpoint}...")
            state_dict = th.load(resume_checkpoint, map_location=dev)#"cpu")
            self.model.load_state_dict(state_dict, strict=False)
            log(f"end loading pretrained model from checkpoint: {resume_checkpoint}...")

        # dist_util.sync_params(self.model.parameters())
        # dist_util.sync_params(self.model.buffers())
        log(f"end synchronizing pretrained model from GPU0 to all GPUs")

    def _load_and_sync_discriminator_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            resume_checkpoint = bf.join(bf.dirname(resume_checkpoint), f"d_model{self.resume_step:06}.pt")
            # if dist.get_rank() == 0:
            if os.path.exists(resume_checkpoint):
                log(f"loading discriminator model from checkpoint: {resume_checkpoint}...")
                #try:
                #if self.args.map_location == 'cuda':
                #    state_dict = dist_util.load_state_dict(
                #        resume_checkpoint, map_location=dist_util.dev()
                #    )
                #else:
                state_dict = th.load(resume_checkpoint, map_location="cpu")
                self.discriminator.load_state_dict(state_dict)
                log(f"end loading discriminator model from checkpoint: {resume_checkpoint}...")

        # dist_util.sync_params(self.discriminator.parameters())
        # dist_util.sync_params(self.discriminator.buffers())
        # log(f"end synchronizing discriminator from GPU0 to all GPUs")

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        if self.args.sanity_check:
            log(f"{rate} ema param before overriding: ", ema_params[1].reshape(-1)[:3])
        main_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # if dist.get_rank() == 0:
            log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            if self.args.map_location == 'cuda':
                state_dict = th.load(ema_checkpoint, map_location= dev)#"cpu")
            else:
                state_dict = load_state_dict(
                    ema_checkpoint, map_location='cpu'
                )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
            log(f"end loading EMA from checkpoint: {ema_checkpoint}...")

        # dist_util.sync_params(ema_params)
        log(f"end synchronizing EMA from GPU0 to all GPUs")
        if self.args.sanity_check:
            log(f"{rate} ema param after overriding: ", ema_params[1].reshape(-1)[:3])

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            #if self.args.map_location == 'cuda':
            #    state_dict = dist_util.load_state_dict(
            #        opt_checkpoint, map_location=dist_util.dev()
            #    )
            #else:
            state_dict = th.load(opt_checkpoint, map_location="cpu")
            self.opt.load_state_dict(state_dict)
            log(f"end loading optimizer state from checkpoint: {opt_checkpoint}")

        if self.args.sanity_check:
            print("opt state dict after overriding: ", self.opt.state_dict()['state'])

    def _load_d_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"d_opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            log(f"loading d_optimizer state from checkpoint: {opt_checkpoint}")
            if os.path.exists(opt_checkpoint):
                if self.args.map_location == 'cuda':
                    state_dict = load_state_dict(
                        opt_checkpoint, map_location=dev
                    )
                else:
                    state_dict = th.load(opt_checkpoint, map_location="cpu")
                self.d_opt.load_state_dict(state_dict)
            log(f"end loading d_optimizer state from checkpoint: {opt_checkpoint}")

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.args.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.args.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logkv("step", self.step + self.resume_step)
        logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def sampling(self, model, sampler, ctm=None, teacher=False, step=-1, num_samples=-1, batch_size=-1, rate=0.999,
                 png=False, resize=True, generator=None, class_generator=None, sample_dir=''):
        if not teacher:
            model.eval()
        if step == -1:
            step = self.args.sampling_steps
        if batch_size == -1:
            batch_size = self.args.sampling_batch

        number = 0
        while num_samples > number:

            with th.no_grad():
                model_kwargs = {}
                if self.args.class_cond:
                    if self.args.train_classes >= 0:
                        classes = th.ones(size=(batch_size,), device=dev, dtype=int) * self.args.train_classes
                        model_kwargs["y"] = classes
                    elif self.args.train_classes == -2:
                        classes = [0, 1, 9, 11, 29, 31, 33, 55, 76, 89, 90, 130, 207, 250, 279, 281, 291, 323, 386, 387,
                                   388, 417, 562, 614, 759, 789, 800, 812, 848, 933, 973, 980]
                        assert batch_size % len(classes) == 0
                        model_kwargs["y"] = th.tensor([x for x in classes for _ in range(batch_size // len(classes))], device=dev)
                    else:
                        if class_generator != None:
                            model_kwargs["y"] = class_generator.randint(0, self.args.num_classes, (batch_size,), device=dev)
                        else:
                            if num_samples == -1:
                                model_kwargs["y"] = self.classes.to(dev)
                            else:
                                model_kwargs["y"] = th.randint(0, self.args.num_classes, size=(batch_size, ), device=dev)
                if generator != None:
                    x_T = generator.randn(*(batch_size, self.args.in_channels, self.args.image_size, self.args.image_size),
                                device=dev) * self.args.sigma_max
                    if self.args.large_log:
                        print("x_T: ", x_T[0][0][0][:3])
                else:
                    x_T = None

                sample = karras_sample(
                    diffusion=self.diffusion,
                    model=model,
                    shape=(batch_size, self.args.in_channels, self.args.image_size, self.args.image_size),
                    steps=step,
                    model_kwargs=model_kwargs,
                    device=dev,
                    clip_denoised=True if teacher else self.args.clip_denoised,
                    sampler=sampler,
                    generator=None,
                    teacher=teacher,
                    ctm=ctm if ctm != None else True if self.args.training_mode.lower() == 'ctm' else False,
                    x_T=x_T if generator != None else self.x_T.to(dev) if num_samples == -1 else None,
                    clip_output=self.args.clip_output,
                    sigma_min=self.args.sigma_min,
                    sigma_max=self.args.sigma_max,
                    train=False,
                )
                if resize:
                    sample = F.interpolate(sample, size=224, mode="bilinear")

                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
                # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                # dist.all_gather(gathered_samples, sample)
                # all_images = [sample.cpu().numpy() for sample in gathered_samples]
                # arr = np.concatenate(all_images, axis=0)
                arr = sample.cpu().numpy()
            # if dist.get_rank() == 0:
                os.makedirs(bf.join(get_blob_logdir(), f"{sample_dir}"), exist_ok=True)
                if self.args.large_log:
                    print(f"saving to {bf.join(get_blob_logdir(), sample_dir)}")
                nrow = int(np.sqrt(arr.shape[0]))
                image_grid = make_grid(th.tensor(arr).permute(0, 3, 1, 2) / 255., nrow, padding=2)
                if num_samples == -1:
                    print("1")
                    with bf.BlobFile(bf.join(get_blob_logdir(), f"{'teacher_' if teacher else ''}sample_{sampler}_sampling_step_{step}_step_{self.step}.png"), "wb") as fout:
                        save_image(image_grid, fout)
                else:
                    print("2")
                    if generator != None:
                        print("3")
                        os.makedirs(bf.join(get_blob_logdir(), sample_dir),
                                    exist_ok=True)
                        np.savez(bf.join(get_blob_logdir(), f"{sample_dir}/sample_{number // arr.shape[0]}.npz"),
                                    arr)
                        if png and number <= 3000:
                            print("4")
                            with bf.BlobFile(bf.join(get_blob_logdir(),
                                                        f"{sample_dir}/sample_{number // arr.shape[0]}.png"), "wb") as fout:
                                save_image(image_grid, fout)
                    else:
                        print("5")
                        r = np.random.randint(100000000)
                        # if self.args.large_log:
                        #     log(f'{dist.get_rank()} number {number}')
                        log(bf.join(get_blob_logdir(), f"{sample_dir}"))
                        os.makedirs(bf.join(get_blob_logdir(), f"{sample_dir}"),
                                    exist_ok=True)
                        np.savez(bf.join(get_blob_logdir(), f"{sample_dir}/sample_{r}.npz"),
                                    arr)
                        if png and number <= 1000:
                            print("6")
                            with bf.BlobFile(bf.join(get_blob_logdir(),
                                                        f"{sample_dir}/sample_{r}.png"), "wb") as fout:
                                save_image(image_grid, fout)

                number += arr.shape[0]
                print(f"{number} number samples complete")
        if not teacher:
            model.train()

    def calculate_similarity_metrics(self, image_path, num_samples=50000, step=1, batch_size=100, rate=0.999, sampler='exact', log=True):
        files = glob.glob(os.path.join(image_path, 'sample*.npz'))
        files.sort()
        count = 0
        psnr = 0
        ssim = 0
        for i, file in enumerate(files):
            images = np.load(file)['arr_0']
            for k in range((images.shape[0] - 1) // batch_size + 1):
                #ref_img = self.ref_images[count + k * batch_size: count + (k + 1) * batch_size]
                if count + batch_size > num_samples:
                    remaining_num_samples = num_samples - count
                else:
                    remaining_num_samples = batch_size
                img = images[k * batch_size: k * batch_size + remaining_num_samples]
                ref_img = self.ref_images[count: count + remaining_num_samples]
                psnr += cv2.PSNR(img, ref_img) * remaining_num_samples
                ssim += SSIM_(img,ref_img,multichannel=True,channel_axis=3,data_range=255) * remaining_num_samples
                count = count + remaining_num_samples
                print(count)
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
        assert count == num_samples
        print(count)
        psnr /= num_samples
        ssim /= num_samples
        assert num_samples % 1000 == 0
        if log:
            log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate} PSNR-{num_samples // 1000}k: {psnr}, SSIM-{num_samples // 1000}k: {ssim}")
        else:
            return psnr, ssim

    def calculate_inception_stats(self, data_name, image_path, num_samples=50000, batch_size=100, device=th.device('cuda')):
        if data_name.lower() == 'cifar10':
            print(f'Loading images from "{image_path}"...')
            mu = th.zeros([self.feature_dim], dtype=th.float64, device=device)
            sigma = th.zeros([self.feature_dim, self.feature_dim], dtype=th.float64, device=device)
            files = glob.glob(os.path.join(image_path, 'sample*.npz'))
            count = 0
            for file in files:
                images = np.load(file)['arr_0']  # [0]#["samples"]
                for k in range((images.shape[0] - 1) // batch_size + 1):
                    mic_img = images[k * batch_size: (k + 1) * batch_size]
                    mic_img = th.tensor(mic_img).permute(0, 3, 1, 2).to(device)
                    features = self.detector_net(mic_img, **self.detector_kwargs).to(th.float64)
                    if count + mic_img.shape[0] > num_samples:
                        remaining_num_samples = num_samples - count
                    else:
                        remaining_num_samples = mic_img.shape[0]
                    mu += features[:remaining_num_samples].sum(0)
                    sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
                    count = count + remaining_num_samples
                    print(count)
                    if count >= num_samples:
                        break
                if count >= num_samples:
                    break
            assert count == num_samples
            print(count)
            mu /= num_samples
            sigma -= mu.ger(mu) * num_samples
            sigma /= num_samples - 1
            mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy()
            return mu, sigma
        else:
            filenames = glob.glob(os.path.join(image_path, '*.npz'))
            imgs = []
            for file in filenames:
                try:
                    img = np.load(file)  # ['arr_0']
                    try:
                        img = img['data']
                    except:
                        img = img['arr_0']
                    imgs.append(img)
                except:
                    pass
            imgs = np.concatenate(imgs, axis=0)
            os.makedirs(os.path.join(image_path, 'single_npz'), exist_ok=True)
            np.savez(os.path.join(os.path.join(image_path, 'single_npz'), f'data'),
                     imgs)  # , labels)
            log("computing sample batch activations...")
            sample_acts = self.evaluator.read_activations(
                os.path.join(os.path.join(image_path, 'single_npz'), f'data.npz'))
            log("computing/reading sample batch statistics...")
            sample_stats, sample_stats_spatial = tuple(self.evaluator.compute_statistics(x) for x in sample_acts)
            with open(os.path.join(os.path.join(image_path, 'single_npz'), f'stats'), 'wb') as f:
                pickle.dump({'stats': sample_stats, 'stats_spatial': sample_stats_spatial}, f)
            with open(os.path.join(os.path.join(image_path, 'single_npz'), f'acts'), 'wb') as f:
                pickle.dump({'acts': sample_acts[0], 'acts_spatial': sample_acts[1]}, f)
            return sample_acts, sample_stats, sample_stats_spatial

    def compute_fid(self, mu, sigma, ref_mu=None, ref_sigma=None):
        if np.array(ref_mu == None).sum():
            ref_mu = self.mu_ref
            assert ref_sigma == None
            ref_sigma = self.sigma_ref
        m = np.square(mu - ref_mu).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma, ref_sigma), disp=False)
        fid = m + np.trace(sigma + ref_sigma - s * 2)
        fid = float(np.real(fid))
        return fid

    def calculate_inception_stats_npz(self, image_path, num_samples=50000, step=1, batch_size=100, device=th.device('cuda'),
                                      rate=0.999):
        print(f'Loading images from "{image_path}"...')
        mu = th.zeros([self.feature_dim], dtype=th.float64, device=device)
        sigma = th.zeros([self.feature_dim, self.feature_dim], dtype=th.float64, device=device)

        files = glob.glob(os.path.join(image_path, 'sample*.npz'))
        count = 0
        for file in files:
            images = np.load(file)['arr_0']  # [0]#["samples"]
            for k in range((images.shape[0] - 1) // batch_size + 1):
                mic_img = images[k * batch_size: (k + 1) * batch_size]
                mic_img = th.tensor(mic_img).permute(0, 3, 1, 2).to(device)
                features = self.detector_net(mic_img, **self.detector_kwargs).to(th.float64)
                if count + mic_img.shape[0] > num_samples:
                    remaining_num_samples = num_samples - count
                else:
                    remaining_num_samples = mic_img.shape[0]
                mu += features[:remaining_num_samples].sum(0)
                sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
                count = count + remaining_num_samples
                log(count)
            if count >= num_samples:
                break
        assert count == num_samples
        print(count)
        mu /= num_samples
        sigma -= mu.ger(mu) * num_samples
        sigma /= num_samples - 1
        mu = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()

        m = np.square(mu - self.mu_ref).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma, self.sigma_ref), disp=False)
        fid = m + np.trace(sigma + self.sigma_ref - s * 2)
        fid = float(np.real(fid))
        assert num_samples % 1000 == 0
        log(f"{self.step}-th step exact sampler (NFE {step}) EMA {rate} FID-{num_samples // 1000}k: {fid}")

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

import nvidia_smi
import functools
import gc
import datetime


def log_loss_dict(losses):
    for key, values in losses.items():
        logkv_mean(f"{key} mean", values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        logkv_mean(f"{key} std", values.std().item())
        #for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #    quartile = int(4 * sub_t / diffusion.num_timesteps)
        #    logkv_mean(f"{key}_q{quartile}", sub_loss)

def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return get_dir()

class CMTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        target_model,
        teacher_model,
        ema_scale_fn,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.training_mode = self.args.training_mode
        self.ema_scale_fn = ema_scale_fn
        self.target_model = target_model
        self.teacher_model = teacher_model
        self.total_training_steps = self.args.total_training_steps

        if target_model:
            if self.args.sanity_check:
                for name, param in self.target_model.named_parameters():
                    log("target model parameter before overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
            self._load_and_sync_ema_parameters_to_target_parameters()
            if self.args.sanity_check:
                # print("doing sanity check ---------------->>>>>>>>>")
                for name, param in self.target_model.named_parameters():
                    log("target model parameter after overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
            self.target_model.requires_grad_(False)
            self.target_model.train()

            if self.args.use_fp16:
                # print("usinig args.usefp16 ----------------->>>>>>>>>>>>>")
                self.target_model_param_groups_and_shapes = get_target_param_groups_and_shapes(
                    self.target_model.named_parameters(), self.model.named_parameters()
                )
                self.target_model_master_params = make_master_params(
                    self.target_model_param_groups_and_shapes
                )
            else:
                # print("In the else part of args.usefp16 ----------------->>>>")
                self.target_model_param_groups_and_shapes = list(self.target_model.named_parameters())
                self.target_model_master_params = list(target_model.parameters())
            for rate, params in zip(self.ema_rate, self.ema_params):
                if rate == 0.999:
                    log(f"loading target model from 0.999 ema...")
                    update_ema(
                        self.target_model_master_params,
                        params,
                        rate=0.0,
                    )
                    if self.args.use_fp16:
                        master_params_to_model_params(
                            self.target_model_param_groups_and_shapes,
                            self.target_model_master_params,
                        )
            if self.args.sanity_check:
                for name, param in self.target_model.named_parameters():
                    log("target model parameter after all: ", param.data.cpu().detach().reshape(-1)[:3])
                    break

        if teacher_model:
            #self._load_and_sync_teacher_parameters()
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
        self.diffusion.teacher_model = teacher_model
        # print("printing the teacher model:::::::::: ", type(teacher_model))
        # print("printing the target_model model:::::::::: ", type(target_model))
        self.global_step = self.step
        self.initial_step = copy.deepcopy(self.step)
        if self.args.gpu_usage:
            nvidia_smi.nvmlInit()
            self.deviceCount = nvidia_smi.nvmlDeviceGetCount()
            self.print_gpu_usage('Before everything')
        # print('self.args.check_dm_performance', self.args.check_dm_performance)
        # exit()check_dm_performance
        # self.args.check_dm_performance = False # -------------------------->>>>>>>>> i have hardcoded this to false. but it was true for uncond
        if self.args.check_dm_performance and False: # ------------>>>>>> it is throwing error at the sampling part
            if not os.path.exists(self.args.dm_sample_path_seed_42):
                self.sampling(model=self.teacher_model, sampler='heun', teacher=True, step=18 if self.args.data_name.lower() == 'cifar10' else 40,
                              num_samples=self.args.eval_num_samples, batch_size=self.args.eval_batch,
                              rate=0.0, ctm=False, png=False, resize=False,
                              generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                              class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                              sample_dir=self.args.dm_sample_path_seed_42)
        print(self.args.data_name.lower())
        if self.args.data_name.lower() == 'cifar10':
            print('Loading Inception-v3 model...')
            detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
            self.detector_kwargs = dict(return_features=True)
            self.feature_dim = 2048
            with open_url(detector_url, verbose=(0 == 0)) as f:
                self.detector_net = pickle.load(f).to(dev)
            with open_url(self.args.ref_path) as f:
                ref = dict(np.load(f))
            self.mu_ref = ref['mu']
            self.sigma_ref = ref['sigma']

    def print_gpu_usage(self, prefix=''):
        for i in range(self.deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            log(
                f"{prefix} |Device {i}| Mem Free: {mem.free / 1024 ** 2:5.2f}MB / {mem.total / 1024 ** 2:5.2f}MB | gpu-util: {util.gpu / 100.0:3.1%} | gpu-mem: {util.memory / 100.0:3.1%} |")


    def _load_and_sync_ema_parameters_to_target_parameters(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            if rate == self.args.start_ema: # 0.999
                log(f"loading target model from {self.args.start_ema} ema...")
                state_dict = self.mp_trainer.master_params_to_state_dict(params)
                self.target_model.load_state_dict(state_dict)
        log(f"end loading target model from {self.args.start_ema} ema...")

    def _load_and_sync_target_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            target_name = name.replace("model", "target_model")
            resume_target_checkpoint = os.path.join(path, target_name)
            if bf.exists(resume_target_checkpoint) == 0:
                log(
                    f"loading target model from checkpoint: {resume_target_checkpoint}..."
                )

                state_dict = load_state_dict(
                    resume_target_checkpoint, map_location=dev
                )
                self.target_model.load_state_dict(state_dict, strict=False)



    def _load_and_sync_teacher_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            teacher_name = name.replace("model", "teacher_model")
            resume_teacher_checkpoint = os.path.join(path, teacher_name)

            if bf.exists(resume_teacher_checkpoint)  == 0:
                log(
                    f"loading teacher model from checkpoint: {resume_teacher_checkpoint}..."
                )
                state_dict = load_state_dict(
                    resume_teacher_checkpoint, map_location=dev
                )
                self.teacher_model.load_state_dict(state_dict)#, strict=False)



    def run_loop(self):
        if self.args.gpu_usage:
            self.print_gpu_usage('Before training')
        saved = False
        while (
            self.step < self.args.lr_anneal_steps
            or self.global_step < self.total_training_steps
        ):
            batch, cond = next(self.data)
            if self.args.large_log:
                print("batch size: ", batch.shape)
                # print("rank: ", dist.get_rank())
            if self.args.intermediate_samples:
                #print("!!: ", self.step, dist.get_rank(), self.args.sample_interval, self.args.training_mode)
                #if dist.get_rank() == 0:
                if self.step == self.initial_step + 10 or (self.step % self.args.sample_interval == self.args.sample_interval - 1):
                    if self.args.training_mode.lower() == 'ctm':
                        if self.args.consistency_weight > 0.:
                            self.sampling(model=self.ddp_model, sampler='exact', num_samples=self.args.sampling_batch, png=True)
                            self.sampling(model=self.ddp_model, sampler='exact', step=2, num_samples=self.args.sampling_batch, png=True)
                            self.sampling(model=self.ddp_model, sampler='exact', step=1, num_samples=self.args.sampling_batch, png=True)
                        else:
                            self.sampling(model=self.ddp_model, sampler='heun', ctm=True, teacher=True, num_samples=self.args.sampling_batch, png=True)
                    elif self.args.training_mode.lower() == 'cd':
                        self.sampling(model=self.ddp_model, sampler='onestep', step=1, num_samples=self.args.sampling_batch, png=True)
                    elif self.args.training_mode.lower() == 'edm':
                        self.sampling(model=self.ddp_model, sampler='heun', step=40, num_samples=self.args.sampling_batch, png=True)
                if self.step == self.initial_step + 10 and self.teacher_model != None:
                    self.sampling(model=self.teacher_model, sampler='heun', ctm=False, teacher=True, num_samples=self.args.sampling_batch, png=True)
            self.run_step(batch, cond)
            if self.args.gpu_usage:
                self.print_gpu_usage('After one step training')
            if self.args.large_log:
                print("mp trainer master parameter after one step update: ", self.mp_trainer.master_params[1].reshape(-1)[:3])
                for name, param in self.model.named_parameters():
                    print("model parameter after one step update: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
                for name, param in self.target_model.named_parameters():
                    print("target model parameter after one step update: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
            if self.args.check_ctm_denoising_ability:
                self.eval(step=18, sampler='heun', teacher=True, ctm=True, rate=0.0)
            if (
                self.global_step
                and self.args.eval_interval != -1
                and self.global_step % self.args.eval_interval == self.args.eval_interval - 1
                #and self.step - self.initial_step > 10
                or self.step == self.args.lr_anneal_steps - 1
                or self.global_step == self.total_training_steps - 1
            ):
                if self.args.gpu_usage:
                    self.print_gpu_usage('Before emptying cache in evaluation 1')
                gc.collect()
                th.cuda.empty_cache()
                if self.args.gpu_usage:
                    self.print_gpu_usage('After emptying cache in evaluation 1')
                model_state_dict = self.model.state_dict()
                if self.args.linear_probing:
                    self.eval(step=18, sampler='heun', teacher=True, ctm=True, rate=0.0,
                              generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                              class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                              delete=True)
                self.evaluation(0.0)
                log('Evaluation with model parameter end')
                for rate, params in zip(self.ema_rate, self.ema_params):
                    if not self.args.compute_ema_fids:
                        if rate != 0.999:
                            continue
                    state_dict = self.mp_trainer.master_params_to_state_dict(params)
                    self.model.load_state_dict(state_dict, strict=False)
                    self.evaluation(rate)
                    log(f'Evaluation with {rate}-EMA model parameter end')
                self.model.load_state_dict(model_state_dict, strict=True)
                del model_state_dict, state_dict
                if self.args.gpu_usage:
                    self.print_gpu_usage('Before emptying cache in evaluation 2')
                gc.collect()
                th.cuda.empty_cache()
                if self.args.gpu_usage:
                    self.print_gpu_usage('After emptying cache in evaluation 2')
            # dist.barrier()
            if (
                    self.global_step
                    and self.args.eval_interval != -1
                    and self.global_step % self.args.save_check_period == self.args.save_check_period - 1
                    #and self.step - self.initial_step > 10000
                    or self.step == self.args.lr_anneal_steps - 1
                    or self.global_step == self.total_training_steps - 1
            ):
                gc.collect()
                th.cuda.empty_cache()
                model_state_dict = self.model.state_dict()
                for rate, params in zip(self.ema_rate, self.ema_params):
                    if rate == 0.999:
                        state_dict = self.mp_trainer.master_params_to_state_dict(params)
                        self.model.load_state_dict(state_dict, strict=False)
                        fid = self.save_check(rate)
                    # if dist.get_rank() == 0:
                        assert fid != None
                        self.fids.append(fid)
                        save_ckpt = (self.fids[-1] == np.min(self.fids))
                        log("FID by iteration (NFE 1, EMA 0.999): ", self.fids)
                self.model.load_state_dict(model_state_dict, strict=True)
                del model_state_dict, state_dict
            # if dist.get_rank() == 0:
                if save_ckpt:
                    self.save(save_full=False)
                gc.collect()
                th.cuda.empty_cache()
            if self.args.large_log:
                print("mp trainer master parameter after sampling: ",
                      self.mp_trainer.master_params[1].reshape(-1)[:3])
                for name, param in self.model.named_parameters():
                    print("model parameter after sampling: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
                for name, param in self.target_model.named_parameters():
                    print("target model parameter after sampling: ", param.data.cpu().detach().reshape(-1)[:3])
                    break

            saved = False
            if (
                self.global_step
                and self.args.save_interval != -1
                and self.global_step % self.args.save_interval == 0
            ):
                self.save()
                if self.discriminator != None:
                    self.d_save()
                saved = True
                gc.collect()
                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.global_step % self.args.log_interval == 0:
                dumpkvs()
                log(datetime.datetime.now().strftime("SONY-%Y-%m-%d-%H-%M-%S"))
            if self.args.large_log:
                print("mp trainer master parameter after saving: ",
                      self.mp_trainer.master_params[1].reshape(-1)[:3])
                for name, param in self.model.named_parameters():
                    print("model parameter after saving: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
                for name, param in self.target_model.named_parameters():
                    print("target model parameter after saving: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
                print(f"0.999 ema param after overriding (should be same to the target parameter): ",
                      self.ema_params[0][1].reshape(-1)[:3])

        # Save the last checkpoint if it wasn't already saved.
        if not saved:
            self.save()
            if self.discriminator != None:
                self.d_save()

    def save_check(self, rate):
        if self.args.training_mode.lower() == 'ctm':
            assert rate == 0.999
            #fid = self.eval(step=1, rate=rate, ctm=True, delete=True, out=True)
            fid = self.eval(step=1, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                                  class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                          metric='similarity', delete=True, out=True)
            return fid

    def evaluation(self, rate):
        if self.args.training_mode.lower() == 'ctm':
            if self.args.eval_fid:
                self.eval(step=1, rate=rate, ctm=True, delete=True)
            if self.args.eval_similarity:
                self.eval(step=1, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                                  class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                          metric='similarity', delete=True)
            if self.args.eval_fid:
                self.eval(step=2, rate=rate, ctm=True, delete=True)
            if self.args.eval_similarity:
                self.eval(step=2, rate=rate, ctm=True,
                          generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                          class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                          metric='similarity', delete=True)
            if self.args.compute_ema_fids:
                if self.args.eval_fid:
                    self.eval(step=4, rate=rate, ctm=True, delete=True)
                if self.args.eval_similarity:
                    self.eval(step=4, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                              class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                              metric='similarity', delete=True)
            step = 18 if self.args.data_name.lower() == 'cifar10' else 40
            if self.args.eval_large_nfe:
                if self.args.eval_fid:
                    self.eval(step=step, rate=rate, ctm=True, delete=True)
                if self.args.eval_similarity:
                    self.eval(step=step, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                              class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                              metric='similarity', delete=True)

        elif self.args.training_mode.lower() == 'cm':
            if self.args.eval_fid:
                self.eval(step=1, sampler='onestep', rate=rate, ctm=False, delete=True)

    def run_step(self, batch, cond):
        if self.args.large_log:
            print("mp trainer master parameter before update: ", self.mp_trainer.master_params[1].reshape(-1)[:3])
            for name, param in self.model.named_parameters():
                print("model parameter before update: ", param.data.cpu().detach().reshape(-1)[:3])
                break
            for name, param in self.target_model.named_parameters():
                print("target model parameter before update: ", param.data.cpu().detach().reshape(-1)[:3])
                break
        self.forward_backward(batch, cond)
        if self.discriminator == None:
            took_step = self.mp_trainer.optimize(self.opt)
            if took_step:
                self._update_ema()
                if self.target_model:
                    self._update_target_ema()
                self.step += 1
                self.global_step += 1
        else:
            if self.step % self.args.g_learning_period == 0:
                took_step = self.mp_trainer.optimize(self.opt)
            else:
                took_step = self.d_mp_trainer.optimize(self.d_opt)
            # print(self.step, took_step)
            if took_step:
                if self.step % self.args.g_learning_period == 0:
                    self._update_ema()
                    if self.target_model:
                        self._update_target_ema()
                self.step += 1
                self.global_step += 1
        self._anneal_lr()
        self.log_step()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(
                self.target_model_master_params,
                self.mp_trainer.master_params,
                rate=target_ema,
            )
            if self.args.use_fp16:
                master_params_to_model_params(
                    self.target_model_param_groups_and_shapes,
                    self.target_model_master_params,
                )

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        if self.discriminator != None:
            self.d_mp_trainer.zero_grad()
        num_heun_step = [self.diffusion.get_num_heun_step(num_heun_step=self.args.num_heun_step)]
        if self.args.gan_specific_time:
            gan_num_heun_step = [self.diffusion.get_num_heun_step(num_heun_step=self.args.gan_num_heun_step,
                                                                  heun_step_strategy=self.args.gan_heun_step_strategy)]
        diffusion_training_ = [np.random.rand() < self.args.diffusion_training_frequency]
        gan_training_ = [np.random.rand() < self.args.gan_training_frequency]
        # dist.broadcast_object_list(num_heun_step, 0)
        # if self.args.gan_specific_time:
            # dist.broadcast_object_list(gan_num_heun_step, 0)
        # dist.broadcast_object_list(diffusion_training_, 0)
        # dist.broadcast_object_list(gan_training_, 0)
        num_heun_step = num_heun_step[0]
        if self.args.gan_specific_time:
            gan_num_heun_step = gan_num_heun_step[0]
        else:
            gan_num_heun_step = -1
        diffusion_training_ = diffusion_training_[0]
        gan_training_ = gan_training_[0]

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dev)
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            compute_losses = functools.partial(
                self.diffusion.ctm_losses,
                step=self.step,
                model=self.ddp_model,
                x_start=micro,
                model_kwargs=micro_cond,
                target_model=self.target_model,
                discriminator=self.ddp_discriminator,
                init_step=self.initial_step,
                ctm=True if self.training_mode.lower() == 'ctm' else False,
                num_heun_step=num_heun_step,
                gan_num_heun_step=gan_num_heun_step,
                diffusion_training_=diffusion_training_,
                gan_training_=gan_training_,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                if self.step % self.args.g_learning_period == 0:
                    # with self.ddp_model.no_sync():
                    losses = compute_losses()
                else:
                    # with self.ddp_discriminator.no_sync():
                    losses = compute_losses()

            if 'consistency_loss' in list(losses.keys()):
                # print("Consistency learning")
                loss = self.args.consistency_weight * losses["consistency_loss"].mean()

                if 'd_loss' in list(losses.keys()):
                    if self.args.large_log:
                        print("GAN learning, ", self.args.discriminator_weight, losses['d_loss'].mean())
                    loss = loss + self.args.discriminator_weight * losses['d_loss'].mean()
                if 'denoising_loss' in list(losses.keys()):
                    loss = loss + self.args.denoising_weight * losses['denoising_loss'].mean()
                log_loss_dict({k: v.view(-1) for k, v in losses.items()})
                if self.args.sanity_check:
                    # print("rank: ", dist.get_rank())
                    for name, param in self.model.named_parameters():
                        print("model parameter gradient for current microbatch: ",
                              th.autograd.grad(outputs=(2**self.mp_trainer.lg_loss_scale) * loss, inputs=param, retain_graph=True)[0].reshape(-1)[:3])
                        break
                self.mp_trainer.backward(loss)

            elif 'd_loss' in list(losses.keys()):
                assert self.step % self.args.g_learning_period != 0
                loss = (losses["d_loss"]).mean()
                self.d_mp_trainer.backward(loss)
                if self.args.large_log:
                    for param in self.discriminator.parameters():
                        try:
                            print("discriminator param data, grad: ", param.grad.reshape(-1)[:3])
                        except:
                            print("discriminator param grad: ", param.grad)
                        break

            elif 'denoising_loss' in list(losses.keys()):
                loss = losses['denoising_loss'].mean()
                log_loss_dict({k: v.view(-1) for k, v in losses.items()})
                self.mp_trainer.backward(loss)
            if self.args.sanity_check:
                # print("rank: ", dist.get_rank())
                for name, param in self.model.named_parameters():
                    print("model parameter gradient across all microbatch: ", param.grad.cpu().detach().reshape(-1)[:3])
                    break

    @th.no_grad()
    def eval(self, step=1, sampler='exact', teacher=False, ctm=False, rate=0.999, generator=None, class_generator=None, metric='fid', delete=False, out=False):
        model = self.model
        sample_dir = f"{self.step}_{sampler}_{step}_{rate}"
        if generator != None:
            sample_dir = sample_dir + "_seed_42"
        self.sampling(model=model, sampler=sampler, teacher=teacher, step=step,
                      num_samples=self.args.eval_num_samples, batch_size=self.args.eval_batch,
                      rate=rate, ctm=ctm, png=True, resize=False, generator=generator,
                      class_generator=class_generator, sample_dir=sample_dir)
        gc.collect()
        th.cuda.empty_cache()
    # if dist.get_rank() == 0:
        if self.args.data_name.lower() == 'cifar10':
            if metric == 'fid':
                mu, sigma = self.calculate_inception_stats(self.args.data_name,
                                                            os.path.join(get_blob_logdir(), sample_dir),
                                                            num_samples=self.args.eval_num_samples)
                log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate}"
                            f" FID-{self.args.eval_num_samples // 1000}k: {self.compute_fid(mu, sigma)}")
            if metric == 'similarity':
                mu, sigma = self.calculate_inception_stats(self.args.data_name,
                                                            os.path.join(get_blob_logdir(), sample_dir),
                                                            num_samples=self.args.eval_num_samples)
                log(f"{self.step}-th step {sampler} sampler (NFE {step}) seed 42 EMA {rate}"
                            f" FID-{self.args.eval_num_samples // 1000}k: {self.compute_fid(mu, sigma)}")
            if self.args.check_dm_performance and False:
                log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate}"
                            f" FID-{self.args.eval_num_samples // 1000}k compared with DM: {self.compute_fid(mu, sigma, self.dm_mu, self.dm_sigma)}")
                self.calculate_similarity_metrics(os.path.join(get_blob_logdir(), sample_dir),
                                                    num_samples=self.args.eval_num_samples, step=step, rate=rate)
            if delete:
                shutil.rmtree(os.path.join(get_blob_logdir(), sample_dir))
            if out:
                return self.compute_fid(mu, sigma)
        else:
            sample_acts, sample_stats, sample_stats_spatial = self.calculate_inception_stats(self.args.data_name,
                                                                            bf.join(get_blob_logdir(), sample_dir),
                                                                            num_samples=self.args.eval_num_samples)
            log(f"Inception Score-{self.args.eval_num_samples // 1000}k:", self.evaluator.compute_inception_score(sample_acts[0]))
            log(f"FID-{self.args.eval_num_samples // 1000}k:", sample_stats.frechet_distance(self.ref_stats))
            log(f"sFID-{self.args.eval_num_samples // 1000}k:", sample_stats_spatial.frechet_distance(self.ref_stats_spatial))
            prec, recall = self.evaluator.compute_prec_recall(self.ref_acts[0], sample_acts[0])
            log("Precision:", prec)
            log("Recall:", recall)
            #self.evaluator.sess.close()
            #tf.reset_default_graph()

    def save(self, save_full=True):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
        # if dist.get_rank() == 0:
            log(f"saving model {rate}...")
            if not rate:
                filename = f"model{self.global_step:06d}.pt"
            else:
                filename = f"ema_{rate}_{self.global_step:06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            if not save_full:
                if rate == 0.999:
                    save_checkpoint(rate, params)
            else:
                save_checkpoint(rate, params)
        if save_full:
            log("saving optimizer state...")
        # if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{self.global_step:06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # if dist.get_rank() == 0:
            if self.target_model:
                log("saving target model state")
                filename = f"target_model{self.global_step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.target_model.state_dict(), f)
            if self.teacher_model and self.training_mode == "progdist":
                log("saving teacher model state")
                filename = f"teacher_model{self.global_step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.teacher_model.state_dict(), f)

            # Save model parameters last to prevent race conditions where a restart
            # loads model at step N, but opt/ema state isn't saved for step N.
            save_checkpoint(0, self.mp_trainer.master_params)
            # dist.barrier()

    def d_save(self):
        log("saving d_optimizer state...")
    # if dist.get_rank() == 0:
        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"d_opt{self.global_step:06d}.pt"),
            "wb",
        ) as f:
            th.save(self.d_opt.state_dict(), f)
        with bf.BlobFile(bf.join(get_blob_logdir(), f"d_model{self.global_step:06d}.pt"), "wb") as f:
            th.save(self.d_mp_trainer.master_params_to_state_dict(self.d_mp_trainer.master_params), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        # dist.barrier()

    def log_step(self):
        step = self.global_step
        logkv("step", step)
        logkv("samples", (step + 1) * self.global_batch)

#### logger.py modules starts here

"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

import os
import sys
import shutil
# import os.path as osp
import json
import time
import datetime
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "at")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if hasattr(val, "__float__"):
                valstr = "%-8.3g" % val
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append(
                "| %s%s | %s%s |"
                % (key, " " * (keywidth - len(key)), val, " " * (valwidth - len(val)))
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        maxlen = 30
        return s[: maxlen - 3] + "..." if len(s) > maxlen else s

    def writeseq(self, seq):
        seq = list(seq)
        for (i, elem) in enumerate(seq):
            self.file.write(elem)
            if i < len(seq) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """

    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.step = 1
        prefix = "events"
        path = os.path.join(os.path.abspath(dir), prefix)
        import tensorflow as tf
        from tensorflow.python import pywrap_tensorflow
        from tensorflow.core.util import event_pb2
        from tensorflow.python.util import compat

        self.tf = tf
        self.event_pb2 = event_pb2
        self.pywrap_tensorflow = pywrap_tensorflow
        self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))

    def writekvs(self, kvs):
        def summary_val(k, v):
            kwargs = {"tag": k, "simple_value": float(v)}
            return self.tf.Summary.Value(**kwargs)

        summary = self.tf.Summary(value=[summary_val(k, v) for k, v in kvs.items()])
        event = self.event_pb2.Event(wall_time=time.time(), summary=summary)
        event.step = (
            self.step
        )  # is there any reason why you'd want to specify the step?
        self.writer.WriteEvent(event)
        self.writer.Flush()
        self.step += 1

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None


def make_output_format(format, ev_dir, log_suffix=""):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(os.path.join(ev_dir, "log%s.txt" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(os.path.join(ev_dir, "progress%s.json" % log_suffix))
    elif format == "csv":
        return CSVOutputFormat(os.path.join(ev_dir, "progress%s.csv" % log_suffix))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(os.path.join(ev_dir, "tb%s" % log_suffix))
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    get_current().logkv(key, val)

def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)

def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)

def dumpkvs():
    """
    Write all of the diagnostics from the current iteration
    """
    return get_current().dumpkvs()

def getkvs():
    return get_current().name2val

def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    get_current().log(*args, level=level)

def debug(*args):
    log(*args, level=DEBUG)

def info(*args):
    log(*args, level=INFO)

def warn(*args):
    log(*args, level=WARN)

def error(*args):
    log(*args, level=ERROR)

def set_level(level):
    """
    Set logging threshold on current logger.
    """
    get_current().set_level(level)

def set_comm(comm):
    get_current().set_comm(comm)

def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return get_current().get_dir()

record_tabular = logkv
dump_tabular = dumpkvs

@contextmanager
def profile_kv(scopename):
    logkey = "wait_" + scopename
    tstart = time.time()
    try:
        yield
    finally:
        get_current().name2val[logkey] += time.time() - tstart

def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """

    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profile_kv(n):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name

# ================================================================
# Backend
# ================================================================


def get_current():
    if Logger.CURRENT is None:
        _configure_default_logger()

    return Logger.CURRENT

class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, comm=None):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.comm = comm

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1

    def dumpkvs(self):
        if self.comm is None:
            d = self.name2val
        else:
            d = mpi_weighted_mean(
                self.comm,
                {
                    name: (val, self.name2cnt.get(name, 1))
                    for (name, val) in self.name2val.items()
                },
            )
            if self.comm.rank != 0:
                d["dummy"] = 1  # so we don't get a warning about empty dict
        out = d.copy()  # Return the dict for unit testing purposes
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(d)
        self.name2val.clear()
        self.name2cnt.clear()
        return out

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def set_comm(self, comm):
        self.comm = comm

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))

def get_rank_without_mpi_import():
    # check environment variables here instead of importing mpi4py
    # to avoid calling MPI_Init() when this module is imported
    for varname in ["PMI_RANK", "OMPI_COMM_WORLD_RANK"]:
        if varname in os.environ:
            return int(os.environ[varname])
    return 0

def mpi_weighted_mean(comm, local_name2valcount):
    """
    Copied from: https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/mpi_util.py#L110
    Perform a weighted average over dicts that are each on a different node
    Input: local_name2valcount: dict mapping key -> (value, count)
    Returns: key -> mean
    """
    all_name2valcount = comm.gather(local_name2valcount)
    if comm.rank == 0:
        name2sum = defaultdict(float)
        name2count = defaultdict(float)
        for n2vc in all_name2valcount:
            for (name, (val, count)) in n2vc.items():
                try:
                    val = float(val)
                except ValueError:
                    if comm.rank == 0:
                        warnings.warn(
                            "WARNING: tried to compute mean on non-float {}={}".format(
                                name, val
                            )
                        )
                else:
                    name2sum[name] += val * count
                    name2count[name] += count
        return {name: name2sum[name] / name2count[name] for name in name2sum}
    else:
        return {}

def save_dict_to_file(dic):
    import os, glob
    file_name = glob.glob(os.path.join(dic['out_dir'], 'training_config*.txt'))
    idx = len(file_name)
    f = open(os.path.join(dic['out_dir'], f'training_config_{idx}.txt'),'w')
    dic = dict(sorted(dic.items()))
    for key in list(dic.keys()):
        f.write(f'{key}: {dic[key]}\n')
    f.close()

def configure(args, dir=None, format_strs=None, comm=None, log_suffix=""):
    """
    If comm is provided, average all numerical stats across that comm
    """
    if dir is None:
        dir = os.getenv("SONY_LOGDIR")
    if dir is None:
        dir = os.path.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime("SONY-%Y-%m-%d-%H-%M-%S-%f"),
        )
    assert isinstance(dir, str)
    dir = os.path.expanduser(dir)
    os.makedirs(os.path.expanduser(dir), exist_ok=True)
    save_dict_to_file(vars(args))
    rank = get_rank_without_mpi_import()
    if rank > 0:
        log_suffix = log_suffix + "-rank%03i" % rank

    if format_strs is None:
        if rank == 0:
            format_strs = os.getenv("SONY_LOG_FORMAT", "stdout,log,csv").split(",")
        else:
            format_strs = os.getenv("SONY_LOG_FORMAT_MPI", "log").split(",")
    format_strs = filter(None, format_strs)
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, comm=comm)
    if output_formats:
        log("Logging to %s" % dir)

def _configure_default_logger():
    configure()
    Logger.DEFAULT = Logger.CURRENT

def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log("Reset logger")

@contextmanager
def scoped_configure(dir=None, format_strs=None, comm=None):
    prevlogger = Logger.CURRENT
    configure(dir=dir, format_strs=format_strs, comm=comm)
    try:
        yield
    finally:
        Logger.CURRENT.close()
        Logger.CURRENT = prevlogger



if th.cuda.is_available():
    dev =  th.device("cuda")
    print(dev)
else:
    dev = th.device("cpu")

def main():
    args = create_argparser().parse_args() # this line is not pruned


    configure(args, dir=args.out_dir)

    log("creating data loader...")

    batch_size = args.batch_size
    data = load_data(
        args=args,
        data_name=args.data_name,
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        type=args.type,
    )
    log("creating model and diffusion...")
    # print("args.target_ema_mode: ", args.target_ema_mode)
    # print("args.start_ema: ", args.start_ema)
    # print("args.scale_mode: ", args.scale_mode)
    # print("args.start_scales: ", args.start_scales)
    # print("args.end_scales: ", args.end_scales)
    # print("args.total_training_steps: ", args.total_training_steps)
    # print("args.distill_steps_per_iter: ", args.distill_steps_per_iter)
    ema_scale_fn = create_ema_and_scales_fn( # this part did not run yet
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )

    # Load Feature Extractor
    feature_extractor = load_feature_extractor(args, eval=True)
    
    # Load Discriminator
    discriminator, discriminator_feature_extractor = load_discriminator_and_d_feature_extractor(args)

    # pruned and added till here <<<-------<-<-<

    # Load Model
    model, diffusion = create_model_and_diffusion(args, feature_extractor, discriminator_feature_extractor)
    model.to(dev)
    model.train()
    if args.use_fp16:
        model.convert_to_fp16()

    if len(args.teacher_model_path) > 0 and not args.self_learn:  # path to the teacher score model.
        log(f"loading the teacher model from {args.teacher_model_path}")
        teacher_model, _ = create_model_and_diffusion(args, teacher=True)
        if not args.edm_nn_ncsn and not args.edm_nn_ddpm:
            if args.map_location == 'cuda':
                teacher_model.load_state_dict(
                    load_state_dict(args.teacher_model_path, map_location=dev),
                )
            else:
                teacher_model.load_state_dict(
                    load_state_dict(args.teacher_model_path, map_location='cpu'),
                )
        teacher_model.to(dev)
        teacher_model.eval()

        def filter_(dst_name):
            dst_ = dst_name.split('.')
            for idx, name in enumerate(dst_):
                if '_train' in name:
                    dst_[idx] = ''.join(name.split('_train'))
            return '.'.join(dst_)

        for dst_name, dst in model.named_parameters():
            for src_name, src in teacher_model.named_parameters():
                if dst_name in ['.'.join(src_name.split('.')[1:]), src_name]:
                    dst.data.copy_(src.data)
                    if args.linear_probing:
                        dst.requires_grad = False
                    break
                if args.linear_probing:
                    if filter_(dst_name) in ['.'.join(src_name.split('.')[1:]), src_name]:
                        dst.data.copy_(src.data)
                        break
        teacher_model.requires_grad_(False)
        if args.edm_nn_ncsn:
            model.model.map_noise.freqs = teacher_model.model.model.map_noise.freqs
        if args.use_fp16:
            teacher_model.convert_to_fp16()
    else:
        teacher_model = None

    if args.training_mode != 'edm':
        log("creating the target model")
        target_model, _ = create_model_and_diffusion(args)

        target_model.to(dev)
        target_model.train()
        for dst, src in zip(target_model.parameters(), model.parameters()):
            dst.data.copy_(src.data)

        if args.use_fp16:
            target_model.convert_to_fp16()
        if args.edm_nn_ncsn:
            target_model.model.map_noise.freqs = teacher_model.model.model.map_noise.freqs

    else:
        target_model = None

    log("training...")

    # print('printring the number of para in model:: ', sum(p.numel() for p in model.parameters()))
    # print("Trainable para of the model ::::::::::", sum(p.numel() for p in model.parameters() if p.requires_grad))

    
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        discriminator=discriminator,
        ema_scale_fn=ema_scale_fn,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        args=args,
    ).run_loop()

def create_argparser():
    defaults = dict(data_name='cifar10')
    defaults.update(train_defaults(defaults['data_name']))
    defaults.update(model_and_diffusion_defaults(defaults['data_name']))
    defaults.update(cm_train_defaults(defaults['data_name']))
    defaults.update(ctm_train_defaults(defaults['data_name']))
    defaults.update(ctm_eval_defaults(defaults['data_name']))
    defaults.update(ctm_loss_defaults(defaults['data_name']))
    defaults.update(ctm_data_defaults(defaults['data_name']))
    defaults.update()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
