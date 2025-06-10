"""
Train a diffusion model on images.
"""

import argparse

from cm import logger
from cm.image_datasets import load_data
from cm.script_util import (
    train_defaults,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    ctm_train_defaults,
    ctm_eval_defaults,
    ctm_loss_defaults,
    ctm_data_defaults,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop
import torch.distributed as dist
import copy
import cm.enc_dec_lib as enc_dec_lib
import torch as th
import os
import socket
import blobfile as bf
import io






def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file for single GPU.
    """
    # Directly load the state dictionary from the file
    state_dict = th.load(path, **kwargs)
    return state_dict

def main():
    args = create_argparser().parse_args() # this line is not pruned
    if th.cuda.is_available():
        dev =  th.device("cuda")
    else:
        dev = th.device("cpu")

    logger.configure(args, dir=args.out_dir)

    logger.log("creating data loader...")

    batch_size = args.batch_size
    data = load_data(
        args=args,
        data_name=args.data_name,
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        train_classes=args.train_classes,
        num_workers=args.num_workers,
        type=args.type,
        deterministic=args.deterministic,
    )

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )

    # Load Feature Extractor
    feature_extractor = enc_dec_lib.load_feature_extractor(args, eval=True)
    # Load Discriminator
    discriminator, discriminator_feature_extractor = enc_dec_lib.load_discriminator_and_d_feature_extractor(args)
    # Load Model
    model, diffusion = create_model_and_diffusion(args, feature_extractor, discriminator_feature_extractor)
    model.to(dev)
    model.train()
    if args.use_fp16:
        model.convert_to_fp16()

    if len(args.teacher_model_path) > 0 and not args.self_learn:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
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
        logger.log("creating the target model")
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

    logger.log("training...")

    print('printring the number of para in model:: ', sum(p.numel() for p in model.parameters()))
    print("Trainable para of the model ::::::::::", sum(p.numel() for p in model.parameters() if p.requires_grad))

    
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
