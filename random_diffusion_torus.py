"""Torus-topology variant of RandomDiffusion for tileable texture generation.

Patches near image edges wrap around to the opposite side, producing images
that tile seamlessly in both dimensions.

Usage:
    See sample_torus.py for a complete inference example.
"""
import sys

import torch
import torchvision.transforms as T

from random_diffusion import RandomDiffusion, hann_window, tile, untile, save_tensor_as_png
from random_diffusion_masks import RandomDiffusionMasks
from random_diffusion_masks import hann_window as mask_hann_window
from random_diffusion_masks import tile as mask_tile, untile as mask_untile


def wrap_extract(tensor, i, j, p, pixel_scale=1):
    """Extract a 2p x 2p patch centered at (i, j) with toroidal wrapping.

    Args:
        tensor: shape (..., H, W)
        i, j: center coordinates (in latent space)
        p: half-patch size in latent space
        pixel_scale: coordinate multiplier (1 for latent, 8 for pixel-space masks)
    """
    ci = i * pixel_scale
    cj = j * pixel_scale
    hp = p * pixel_scale
    H, W = tensor.shape[-2], tensor.shape[-1]

    # Fast path: no wrapping needed
    if ci - hp >= 0 and ci + hp <= H and cj - hp >= 0 and cj + hp <= W:
        return tensor[..., ci - hp:ci + hp, cj - hp:cj + hp]

    # Slow path: wrapping via modulo indexing
    rows = torch.arange(ci - hp, ci + hp) % H
    cols = torch.arange(cj - hp, cj + hp) % W
    return tensor[..., rows[:, None], cols[None, :]]


def wrap_write(target, patch, i, j, p, pixel_scale=1):
    """Write a 2p x 2p patch into target at (i, j) with toroidal wrapping.

    Args:
        target: shape (..., H, W) — modified in place
        patch: shape (..., 2p, 2p)
        i, j: center coordinates (in latent space)
        p: half-patch size in latent space
        pixel_scale: coordinate multiplier
    """
    ci = i * pixel_scale
    cj = j * pixel_scale
    hp = p * pixel_scale
    H, W = target.shape[-2], target.shape[-1]

    # Fast path
    if ci - hp >= 0 and ci + hp <= H and cj - hp >= 0 and cj + hp <= W:
        target[..., ci - hp:ci + hp, cj - hp:cj + hp] = patch
        return

    # Slow path
    rows = torch.arange(ci - hp, ci + hp) % H
    cols = torch.arange(cj - hp, cj + hp) % W
    target[..., rows[:, None], cols[None, :]] = patch


class RandomDiffusionTorus(RandomDiffusion):
    """Torus-topology sliding-window diffusion for tileable texture generation.

    Overrides random_crop, __call__, and hann_tile_overlap to wrap at edges.
    The model itself is unchanged — it only sees 64x64 patches.
    """

    def __init__(self, trainer, patch_size=512, sampling_steps=50, cond_scale=3.0, device='cuda:0'):
        # Unwrap VAE safely (handles Accelerator/DDP wrapping)
        if hasattr(trainer, '_get_vae'):
            self.vae = trainer._get_vae().to(device)
        else:
            vae = trainer.vae
            self.vae = (vae.module if hasattr(vae, 'module') else vae).to(device)

        # Unwrap EMA to get the actual diffusion model
        ema = trainer.ema
        if hasattr(ema, 'module'):
            ema = ema.module
        if hasattr(ema, 'ema_model'):
            self.ema_model = ema.ema_model.to(device)
        else:
            self.ema_model = ema.to(device)

        self.sampling_steps = sampling_steps
        self.cond_scale = cond_scale + 1
        self.patch_size = patch_size
        self.device = device

        times = torch.linspace(-1, 1000 - 1, steps=self.sampling_steps + 1).to(device)
        times = list(reversed(times.int().tolist()))
        self.time_pairs = list(zip(times[:-1], times[1:]))

    def random_crop(self, image, i, j, latent=True):
        p = self.patch_size // 16
        if latent:
            return wrap_extract(image, i, j, p, pixel_scale=1)
        else:
            return wrap_extract(image, i, j, p, pixel_scale=8)

    @torch.no_grad()
    def __call__(self, masks):
        b, c, image_h, image_w = masks.shape
        masks = masks.to(self.device)

        latent_h = image_h // 8
        latent_w = image_w // 8

        img_stack = torch.randn(b, 2, 4, latent_h, latent_w).to(self.device)
        times = torch.zeros((1, 1, latent_h, latent_w)).int().to(self.device)

        img0 = torch.randn(b, 4, latent_h, latent_w).to(self.device)
        p = self.patch_size // 16

        while times.float().mean() != (self.sampling_steps - 1):
            sys.stdout.flush()
            random_indices = self.get_value_coordinates(times[0, 0])[0]
            # No clamping — allow any position, wrapping handles boundaries
            i, j = random_indices.tolist()
            print(f"\r Generation {times.float().mean() * 100 / (self.sampling_steps - 1):.2f}%", end="")

            sub_img = self.random_crop(img0, i, j)
            sub_img_stack = self.random_crop(img_stack, i, j)
            sub_time = self.random_crop(times, i, j)
            sub_mask = self.random_crop(masks, i, j, latent=False)

            if sub_time.float().mean() != (self.sampling_steps - 1):
                sub_img = self.sample_one(sub_img, sub_img_stack, sub_mask, sub_time)

                mask_changed = torch.where(sub_time == sub_time.min(), 1, 0).to(self.device)

                # Write back with wrapping
                wrap_write(img0, sub_img, i, j, p)

                new_stack = sub_img * mask_changed + wrap_extract(img_stack[:, 1], i, j, p) * (mask_changed == 0)
                wrap_write(img_stack[:, 1], new_stack, i, j, p)

                new_times = torch.where(sub_time == sub_time.min(), sub_time + 1, sub_time)
                wrap_write(times, new_times, i, j, p)

                if torch.all(times == times.max()):
                    img_stack[:, 0] = img_stack[:, 1]

        print()  # newline after progress
        return img0

    def hann_tile_overlap(self, z):
        """Decode latent to pixel space with torus-aware Hann window blending.

        Uses torch.roll instead of edge truncation so the blending wraps around.
        Supports non-square shapes — height and width are handled independently.
        """
        b, c, h, w = z.shape
        latent_patch = self.patch_size // 8
        assert h % latent_patch == 0 and w % latent_patch == 0, \
            f'Latent size {h}x{w} must be a multiple of {latent_patch}'

        windows = hann_window(self.patch_size)
        z_scaled = z.clone() * 50
        p = self.patch_size
        p16 = self.patch_size // 16

        # Pass 1: Grid-aligned decode
        img = self.decoding_tiled_image(z_scaled, (b, 3, h * 8, w * 8))

        # Pass 2: Column-shifted (fixes vertical seams)
        z_v = torch.roll(z_scaled, shifts=-p16, dims=-1)
        img_v = self.decoding_tiled_image(z_v, (b, 3, h * 8, w * 8))
        img_v = torch.roll(img_v, shifts=p16 * 8, dims=-1)

        # Pass 3: Row-shifted (fixes horizontal seams)
        z_h = torch.roll(z_scaled, shifts=-p16, dims=-2)
        img_h = self.decoding_tiled_image(z_h, (b, 3, h * 8, w * 8))
        img_h = torch.roll(img_h, shifts=p16 * 8, dims=-2)

        # Pass 4: Both-shifted (fixes cross seams)
        z_c = torch.roll(z_scaled, shifts=(-p16, -p16), dims=(-2, -1))
        img_c = self.decoding_tiled_image(z_c, (b, 3, h * 8, w * 8))
        img_c = torch.roll(img_c, shifts=(p16 * 8, p16 * 8), dims=(-2, -1))

        # Blend with Hann windows (full tiling, rolled to align with seam positions)
        b_px, c_px, h_px, w_px = img.shape

        wv = windows['vertical'].repeat(b_px, c_px, h_px // p, w_px // p).to(self.device)
        wv = torch.roll(wv, shifts=p // 2, dims=-1)

        wh = windows['horizontal'].repeat(b_px, c_px, h_px // p, w_px // p).to(self.device)
        wh = torch.roll(wh, shifts=p // 2, dims=-2)

        wc = windows['center'].repeat(b_px, c_px, h_px // p, w_px // p).to(self.device)
        wc = torch.roll(wc, shifts=(p // 2, p // 2), dims=(-2, -1))

        img = img * (1 - wv) + img_v * wv
        img = img * (1 - wh) + img_h * wh
        img = img * (1 - wc) + img_c * wc

        return img


class RandomDiffusionMasksTorus(RandomDiffusionMasks):
    """Torus-topology variant of RandomDiffusionMasks for tileable mask generation.

    Same wrapping logic as RandomDiffusionTorus, but for the mask diffusion model.
    The mask model uses a different img_stack structure (sampling_steps frames
    instead of 2) and runs partially on CPU.
    """

    def __init__(self, trainer, patch_size=512, sampling_steps=50, cond_scale=3.0, device='cuda:0'):
        # Unwrap VAE safely
        if hasattr(trainer, '_get_vae'):
            self.vae = trainer._get_vae().to(device)
        else:
            vae = trainer.vae
            self.vae = (vae.module if hasattr(vae, 'module') else vae).to(device)

        # Unwrap EMA
        ema = trainer.ema
        if hasattr(ema, 'module'):
            ema = ema.module
        if hasattr(ema, 'ema_model'):
            self.ema_model = ema.ema_model.to(device)
        else:
            self.ema_model = ema.to(device)

        self.sampling_steps = sampling_steps
        self.cond_scale = cond_scale + 1
        self.patch_size = patch_size
        self.device = device

        times = torch.linspace(-1, 1000 - 1, steps=self.sampling_steps + 1)
        times = list(reversed(times.int().tolist()))
        self.time_pairs = list(zip(times[:-1], times[1:]))

    def random_crop(self, image, i, j, latent=True):
        p = self.patch_size // 16
        if latent:
            return wrap_extract(image, i, j, p, pixel_scale=1)
        else:
            return wrap_extract(image, i, j, p, pixel_scale=8)

    @torch.no_grad()
    def __call__(self, masks):
        b, c, image_h, image_w = masks.shape

        latent_h = image_h // 8
        latent_w = image_w // 8

        img_stack = torch.randn(b, self.sampling_steps, 4, latent_h, latent_w)
        times = torch.zeros((b, 1, latent_h, latent_w)).int()

        img0 = torch.randn(b, 4, latent_h, latent_w)
        p = self.patch_size // 16

        while times.float().mean() != (self.sampling_steps - 1):
            sys.stdout.flush()
            random_indices = self.get_value_coordinates(times[0, 0])[0]
            # No clamping — wrapping handles boundaries
            i, j = random_indices.tolist()
            print(f"\r Mask generation {times.float().mean() * 100 / (self.sampling_steps - 1):.2f}%", end="")

            sub_img = self.random_crop(img0, i, j)
            sub_img_stack = self.random_crop(img_stack, i, j)
            sub_time = self.random_crop(times, i, j)
            sub_mask = self.random_crop(masks, i, j, latent=False)

            if sub_time.float().mean() != (self.sampling_steps - 1):
                sub_img = self.sample_one(sub_img, sub_img_stack, sub_mask, sub_time)

                mask_changed = torch.where(sub_time == sub_time.min(), 1, 0)

                # Write back with wrapping
                wrap_write(img0, sub_img, i, j, p)

                t_idx = sub_time.min() + 1
                old_stack = wrap_extract(img_stack[:, t_idx], i, j, p)
                new_stack = sub_img * mask_changed + old_stack * (mask_changed == 0)
                wrap_write(img_stack[:, t_idx], new_stack, i, j, p)

                new_times = torch.where(sub_time == sub_time.min(), sub_time + 1, sub_time)
                wrap_write(times, new_times, i, j, p)

        print()  # newline after progress
        return img0

    def hann_tile_overlap(self, z):
        """Decode mask latent to pixel space with torus-aware Hann blending.

        Supports non-square shapes — height and width are handled independently.
        """
        b, c, h, w = z.shape
        latent_patch = self.patch_size // 8
        assert h % latent_patch == 0 and w % latent_patch == 0, \
            f'Latent size {h}x{w} must be a multiple of {latent_patch}'

        windows = mask_hann_window(self.patch_size)
        z_scaled = z.clone() * 50
        p = self.patch_size
        p16 = self.patch_size // 16

        # Pass 1: Grid-aligned decode
        img = self.decoding_tiled_image(z_scaled, (b, 3, h * 8, w * 8))

        # Pass 2: Column-shifted
        z_v = torch.roll(z_scaled, shifts=-p16, dims=-1)
        img_v = self.decoding_tiled_image(z_v, (b, 3, h * 8, w * 8))
        img_v = torch.roll(img_v, shifts=p16 * 8, dims=-1)

        # Pass 3: Row-shifted
        z_h = torch.roll(z_scaled, shifts=-p16, dims=-2)
        img_h = self.decoding_tiled_image(z_h, (b, 3, h * 8, w * 8))
        img_h = torch.roll(img_h, shifts=p16 * 8, dims=-2)

        # Pass 4: Both-shifted
        z_c = torch.roll(z_scaled, shifts=(-p16, -p16), dims=(-2, -1))
        img_c = self.decoding_tiled_image(z_c, (b, 3, h * 8, w * 8))
        img_c = torch.roll(img_c, shifts=(p16 * 8, p16 * 8), dims=(-2, -1))

        # Blend with Hann windows
        b_px, c_px, h_px, w_px = img.shape

        wv = windows['vertical'].repeat(b_px, c_px, h_px // p, w_px // p)
        wv = torch.roll(wv, shifts=p // 2, dims=-1)

        wh = windows['horizontal'].repeat(b_px, c_px, h_px // p, w_px // p)
        wh = torch.roll(wh, shifts=p // 2, dims=-2)

        wc = windows['center'].repeat(b_px, c_px, h_px // p, w_px // p)
        wc = torch.roll(wc, shifts=(p // 2, p // 2), dims=(-2, -1))

        img = img * (1 - wv) + img_v * wv
        img = img * (1 - wh) + img_h * wh
        img = img * (1 - wc) + img_c * wc

        return img
