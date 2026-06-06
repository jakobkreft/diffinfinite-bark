"""Generate a large-scale tileable semantic mask using the trained mask model.

Usage:
    python sample_torus_mask.py --image_size 2048 --label 1
    python sample_torus_mask.py --image_size 2048 --label 0 --output_path bark_only_mask.png
    python sample_torus_mask.py --image_size 4096 --label 1 --sampling_steps 100
"""
import os
import fire
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from dm_masks import Unet as MaskUnet, GaussianDiffusion as MaskGD, Trainer as MaskTrainer
from random_diffusion_torus import RandomDiffusionMasksTorus
from utils.mask_modules import CleanMask


def main(
    image_size: int = 2048,
    image_h: int = None,
    image_w: int = None,
    label: int = 1,
    output_path: str = 'generated_mask.png',
    # Mask model config (must match training)
    dim: int = 64,
    num_classes: int = 3,
    num_labels: int = 3,
    dim_mults: str = '1,2,4,8',
    channels: int = 4,
    resnet_block_groups: int = 2,
    block_per_layer: int = 2,
    mask_size: int = 128,
    timesteps: int = 1000,
    sampling_timesteps: int = 250,
    # Checkpoint
    results_folder: str = './results/bark_masks_10k',
    milestone: int = 12,
    # Generation
    patch_size: int = 128,
    sampling_steps: int = 250,
    cond_scale: float = 0.0,
    device: str = 'cuda:0',
):
    dim_mults_list = [int(x) for x in dim_mults.split(',')]

    # Build mask model
    unet = MaskUnet(
        dim=dim, num_classes=num_classes, dim_mults=dim_mults_list,
        channels=channels, resnet_block_groups=resnet_block_groups,
        block_per_layer=block_per_layer,
    )
    model = MaskGD(
        unet, image_size=mask_size // 8, timesteps=timesteps,
        sampling_timesteps=sampling_timesteps, loss_type='l2',
    )
    trainer = MaskTrainer(
        model, train_batch_size=4, train_lr=1e-4, train_num_steps=60000,
        save_and_sample_every=99999, num_workers=0, num_labels=num_labels,
        results_folder=results_folder,
    )

    print(f"Loading mask checkpoint {milestone} from {results_folder}...")
    trainer.load(milestone)

    # Resolve final output dimensions (height, width)
    h_out = image_h if image_h is not None else image_size
    w_out = image_w if image_w is not None else image_size

    # Guide mask: coarse label map (the mask model is conditioned on this)
    guide = torch.ones((1, 1, h_out // 4, w_out // 4)) * label

    print(f"Generating tileable mask ({h_out}x{w_out}, label={label})...")
    sampler = RandomDiffusionMasksTorus(
        trainer, patch_size=patch_size, sampling_steps=sampling_steps,
        cond_scale=cond_scale, device=device,
    )

    with torch.no_grad():
        z = sampler(guide.to(device))

        # z may be (B,C,H,W) or (C,H,W) depending on batch dim
        if z.dim() == 3:
            z = z.unsqueeze(0)

        print("Decoding with Hann tile overlap...")
        mask_decoded = sampler.hann_tile_overlap(z)

        # Convert to discrete classes
        mask_rgb = torch.round((num_labels - 1) * mask_decoded[0])
        mask_gray = torch.round(torch.mean(mask_rgb, dim=0, keepdim=True)) / (num_labels - 1)

        # Clean and upsample to target size (supports rectangular)
        # CleanMask resizes to (ups_mask_size, ups_mask_size); patch its resize to (w,h)
        clean_module = CleanMask(num_labels=num_labels, ups_mask_size=max(h_out, w_out))
        clean = clean_module(mask_gray)
        clean = clean.resize((w_out, h_out), Image.NEAREST)
        mask_np = np.array(clean, dtype=np.uint8)

    # Save as discrete mask (values 0, 1, 2)
    Image.fromarray(mask_np).save(output_path)
    print(f"Saved mask to {output_path}: {mask_np.shape}, classes={np.unique(mask_np).tolist()}")

    # Save a visible version (scaled to 0-255 for viewing)
    vis_path = output_path.replace('.png', '_visible.png')
    vis = (mask_np.astype(np.float32) / max(1, num_labels - 1) * 255).astype(np.uint8)
    Image.fromarray(vis).save(vis_path)
    print(f"Saved visible mask to {vis_path}")

    # Save 2x2 tiled preview
    preview_path = output_path.replace('.png', '_2x2.png')
    tiled = np.tile(vis, (2, 2))
    Image.fromarray(tiled).save(preview_path)
    print(f"Saved 2x2 tiled preview to {preview_path}")


if __name__ == '__main__':
    fire.Fire(main)
