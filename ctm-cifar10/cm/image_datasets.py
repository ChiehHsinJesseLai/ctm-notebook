import math
import random

from PIL import Image
import blobfile as bf

import numpy as np
from torch.utils.data import DataLoader, Dataset
import os


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
    
    all_files = _list_image_files_recursively(data_dir, type=type)
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
        if self.type == 'npy':
            print("i am inside self.tupe == npy")
            data = np.load(path)
            out_dict = {}
            if self.local_classes is not None:
                print("i am inside self.local_classes is not None")
                out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            return data, out_dict
        else:
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

