import torch as th
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid, save_image
import blobfile as bf
import os
import torchvision
from torchvision.transforms import Normalize
import torch.nn as nn

from utils import *


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


### Creating model and diffusion

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

if th.cuda.is_available():
    dev =  th.device("cuda")
    print(dev)
else:
    dev = th.device("cpu")


class KarrasDenoiser:
    """This class consists of all the proposed concepts related to CTM"""
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

    def get_outer_scalings(self, t, s=None):
        # if outer_parametrization == 'euler':

        c_skip = s / t
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
        t = self.args.sigma_max ** (1 / self.args.rho) + ind / (self.args.start_scales - 1) * (
                self.args.sigma_min ** (1 / self.args.rho) - self.args.sigma_max ** (1 / self.args.rho)
        )
        t = t ** self.args.rho
        return t

    def get_num_heun_step(self, start_scales=-1, num_heun_step=-1, num_heun_step_random=None, heun_step_strategy='', time_continuous=None):
        # Random heun steps = True, Time continuous = False, Heun step strategy = uniform
        num_heun_step = np.random.randint(1,1+self.args.num_heun_step)
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
                fake = self.get_ctm_estimate(x_t, t, s, model, target_model, ctm,
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

    def get_ctm_estimate(self, x_t, t, s, model, target_model, ctm, outer_type, **model_kwargs):
        
        # Target matching = False
        # inner type = model
        # Training mode = CTM
        # large log = False
        _, estimate = self.get_denoised_and_G(model, x_t, t, s=s, ctm=ctm, **model_kwargs)
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
        # Large log = False
        # inner type = target sg
        with th.no_grad():
            _, target = self.get_denoised_and_G(target_model, x_t_dt, t_dt, s=s, ctm=ctm, **model_kwargs)
            if self.args.training_mode == 'ctm':
                _, target = self.get_denoised_and_G(target_model, target, s, s=th.ones_like(s) * self.args.sigma_min, ctm=ctm, **model_kwargs)

        return target.detach()


    def get_denoised_and_G(self, model, x_t, t, s=None, ctm=False, teacher=False, **model_kwargs):
        rescaled_t = self.rescaling_t(t)
        if s != None:
            rescaled_s = self.rescaling_t(s)
        else:
            rescaled_s = None
        c_in = append_dims(self.get_c_in(t), x_t.ndim)
        model_output = model(c_in * x_t, rescaled_t, s=rescaled_s, teacher=teacher, **model_kwargs)

        c_skip, c_out = [
            append_dims(x, x_t.ndim)
            for x in self.get_edm_scalings(t)
        ]
        denoised = c_out * model_output + c_skip * x_t

        if s == None:
            return denoised, denoised

        c_skip, c_out = [
            append_dims(x, x_t.ndim)
            for x in self.get_outer_scalings(t, s)
        ]
        G_theta = c_out * denoised + c_skip * x_t

        return denoised, G_theta

    def get_CTM_loss(self, estimate, target, weights, step):
        # CTM loss = LPIPS
        if estimate.shape[-2] < 256:
            estimate = F.interpolate(estimate, size=224, mode="bilinear")
            target = F.interpolate(
                target, size=224, mode="bilinear"
            )
        consistency_loss = (self.feature_extractor(
            (estimate + 1) / 2.0,
            (target + 1) / 2.0, ) * weights)
        return consistency_loss

    def get_DSM_loss(self, model, x_start, model_kwargs, consistency_loss, step, init_step):
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

        # Getting the timesteps for s and t
        dims = x_start.ndim
        s = None
        terms = {}

        # Get the number of heun step: 
        # This step selects random heun step OR timestep t
        num_heun_step = self.get_num_heun_step(num_heun_step=self.args.num_heun_step)
        
        # Get the indexes for timestep
        indices, _ = self.schedule_sampler.sample_t(x_start.shape[0], x_start.device, num_heun_step,
                                                    self.args.time_continuous)
        # Get the EDM mapped t timestep 
        t = self.get_t(indices)

        # Get the EDM mapped u timestep --> only used with teacher
        t_dt = self.get_t(indices + num_heun_step)
        if ctm:
            # Get the index for s timestep
            new_indices = self.schedule_sampler.sample_s(self.args, x_start.shape[0], x_start.device, indices,
                                                         num_heun_step, self.args.time_continuous,
                                                         N=self.args.start_scales)
            # Get the EDM mapped s timestep 
            s = self.get_t(new_indices)
        
        # Add noise to image
        x_t = x_start + noise * append_dims(t, dims)
        
        dropout_state = th.get_rng_state()
        th.set_rng_state(dropout_state)

        # Get the output/estimate of student model
        if self.args.ctm_training:
            # The student model tries to predict denoisified version of x_t at timestep s given timestep t
            ctm_estimate = self.get_ctm_estimate(x_t, t, s, model, target_model, ctm=ctm,
                                                 outer_type=self.args.ctm_estimate_outer_type,
                                                 **model_kwargs)
        
        # If Adversarial training is enabled (gan_training = True)
        # we update Generator only at certain frequencies decided by argument g_learning_period
        # Generator update stage
        #   If Gan training is enabled
        #       loss = CTM loss + DSM loss + Generator loss(If gan training is enabled)
        #   else
        #       loss = CTM loss + DSM loss
        if step % self.args.g_learning_period == 0 or not self.args.gan_training:
            x_t_dt = self.heun_solver(target_model, x_t, indices, dims, t, t_dt, ctm=ctm, num_step=num_heun_step,
                                        **model_kwargs).detach()
            ctm_target = self.get_ctm_target(x_t_dt, t_dt, s, model, target_model, ctm=ctm,
                                                inner_type=self.args.ctm_target_inner_type, **model_kwargs)

            snrs = self.get_snr(t)
            weights = get_weightings(self.args.weight_schedule, snrs, self.args.sigma_data, t, s, self.args.weight_schedule_multiplier)

            terms["consistency_loss"] = self.get_CTM_loss(ctm_estimate, ctm_target, weights, step - init_step,)

            terms['denoising_loss'] = self.get_DSM_loss(model, x_start, model_kwargs,
                                                                terms["consistency_loss"] if self.args.ctm_training else None,
                                                                step, init_step)

            if self.args.gan_training and step - init_step >= self.args.discriminator_start_itr:
                if gan_training_:
                    gan_x_t, gan_t, gan_t_dt, gan_s, _, _ = self.get_gan_time(x_start, noise, x_t, t, t_dt, s, indices,
                                                                              num_heun_step, gan_num_heun_step)
                    gan_fake = self.get_gan_fake(ctm_estimate, gan_x_t, gan_t, gan_t_dt, gan_s, model, target_model, ctm,
                                                 step - init_step, **model_kwargs)
                    terms['d_loss'] = self.get_GAN_loss(model, fake=gan_fake,
                                                                  consistency_loss=terms["consistency_loss"],
                                                                  discriminator=discriminator,
                                                                  step=step, init_step=init_step)
        # Discriminator update stage
        # loss = Adversarial GAN loss
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

