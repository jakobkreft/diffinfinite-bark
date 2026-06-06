"""Generate tileable textures using DiffInfinite with torus topology.

The output image tiles seamlessly in both dimensions.

Usage:
    python sample_torus.py --mask_path path/to/tileable_mask.png

    # Use a specific checkpoint
    python sample_torus.py --mask_path mask.png --results_folder ./results/bark_200k --milestone 5

    # Adjust generation parameters
    python sample_torus.py --mask_path mask.png --sampling_steps 100 --cond_scale 3.0

    # Generate a uniform mask (all bark, no knots/defects) at 2048x2048
    python sample_torus.py --uniform_size 2048 --uniform_class 0

Note: Mask dimensions must be multiples of patch_size (default 512).
      E.g., 2048x2048, 2560x2560, 3072x3072, 3584x3584, 4096x4096.
"""
import os
import fire
import torch
import numpy as np
from PIL import Image

from dm import Unet, GaussianDiffusion, Trainer
from random_diffusion_torus import RandomDiffusionTorus
from random_diffusion import save_tensor_as_png


def main(
    mask_path: str = None,
    output_path: str = 'torus_output.png',
    # Model config (must match training)
    dim: int = 256,
    num_classes: int = 4,
    dim_mults: str = '1,2,4',
    channels: int = 4,
    resnet_block_groups: int = 2,
    block_per_layer: int = 2,
    image_size: int = 512,
    timesteps: int = 1000,
    sampling_timesteps: int = 250,
    # Checkpoint
    results_folder: str = './results/bark_200k',
    milestone: int = 5,
    # Generation
    patch_size: int = 512,
    sampling_steps: int = 250,
    cond_scale: float = 3.0,
    device: str = 'cuda:0',
    # Uniform mask generation (alternative to mask_path)
    uniform_size: int = None,
    uniform_h: int = None,
    uniform_w: int = None,
    uniform_class: int = 0,
):
    dim_mults_list = [int(x) for x in dim_mults.split(',')]

    # Build model
    unet = Unet(
        dim=dim, num_classes=num_classes, dim_mults=dim_mults_list,
        channels=channels, resnet_block_groups=resnet_block_groups,
        block_per_layer=block_per_layer,
    )
    model = GaussianDiffusion(
        unet, image_size=image_size // 8, timesteps=timesteps,
        sampling_timesteps=sampling_timesteps, loss_type='l2',
    )
    trainer = Trainer(
        model, train_batch_size=2, train_lr=1e-4, train_num_steps=200000,
        save_and_sample_every=99999, num_workers=0,
        results_folder=results_folder,
    )

    print(f"Loading checkpoint {milestone} from {results_folder}...")
    trainer.load(milestone)

    # Prepare mask
    if mask_path is not None:
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        h, w = mask.shape
    elif uniform_size is not None or (uniform_h is not None and uniform_w is not None):
        h = uniform_h if uniform_h is not None else uniform_size
        w = uniform_w if uniform_w is not None else uniform_size
        mask = np.full((h, w), uniform_class, dtype=np.uint8)
    else:
        print("Error: provide either --mask_path or --uniform_size")
        return

    assert h % patch_size == 0 and w % patch_size == 0, \
        f"Mask dimensions ({h}x{w}) must be multiples of patch_size ({patch_size}). " \
        f"Nearest valid sizes: {(h // patch_size) * patch_size}x{(w // patch_size) * patch_size} " \
        f"or {((h // patch_size) + 1) * patch_size}x{((w // patch_size) + 1) * patch_size}"

    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
    print(f"Mask: {h}x{w}, classes: {np.unique(mask).tolist()}")

    # Generate
    sampler = RandomDiffusionTorus(
        trainer, patch_size=patch_size, sampling_steps=sampling_steps,
        cond_scale=cond_scale, device=device,
    )

    print(f"Generating tileable texture ({h}x{w})...")
    with torch.no_grad():
        z = sampler(mask_tensor)
        print("Decoding with Hann tile overlap...")
        img = sampler.hann_tile_overlap(z)

    save_tensor_as_png(img[0], output_path)
    print(f"Saved tileable texture to {output_path}")

    # Also save a 2x2 tiled preview
    preview_path = output_path.replace('.png', '_2x2.png')
    img_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    tiled = np.tile(img_np, (2, 2, 1))
    Image.fromarray(tiled).save(preview_path)
    print(f"Saved 2x2 tiled preview to {preview_path}")


if __name__ == '__main__':
    fire.Fire(main)
