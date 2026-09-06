"""Visualize the denoising trajectory of a trained DiffInfinite-Bark model.

Reproduces the standard diffusion-paper methodology figure: a row of decoded
snapshots evenly spread across DDIM sampling, showing how the model's
estimate of the final image emerges from Gaussian noise.

Two visualization modes:

- 'pred_x0' (default, and what you actually want): at each snapshot decode
  the model's PREDICTED clean latent x_0 (best guess of the final image
  given what the model sees now). This looks coherent and increasingly
  refined from step 1 — the canonical paper-style trajectory.

- 'raw_xt': decode the actual noisy latent x_t at each snapshot. Looks like
  pure VAE-decoded garbage until the last ~20% of steps, because the VAE
  decoder was trained on clean latents and behaves poorly on noisy inputs.
  This is what the first version of the script did.

- 'both': stack both rows in the output figure so you can see the contrast
  side by side. Great for the thesis's methodology section.

Snapshots are evenly spaced across the ACTUAL DDIM sampling steps (so with
sampling_steps=250 and n_snapshots=10, snapshots are taken at steps
0, 27, 54, 82, ..., 249 — corresponding to actual DDIM timesteps that get
visited). If you pass --snapshot_t_values '1000,900,800,...', each target
is mapped to the nearest actual DDIM step (equality checks like the old
script produced NO intermediate snapshots because e.g. t=800 was never
visited by the sampler).

Outputs:
    <output_dir>/denoising_trajectory_milestone{N}.png   <-- grid figure
    <output_dir>/denoising_trajectory_milestone{N}_annotated.png
    <output_dir>/denoising_trajectory_milestone{N}.gif   <-- if --make_gif

Usage:
    python scripts/visualize_denoising.py --results_folder ./results/bark_1024_aug_200k --milestone 40
    python scripts/visualize_denoising.py --milestone 40 --n_snapshots 10
    python scripts/visualize_denoising.py --milestone 40 --mode both
    python scripts/visualize_denoising.py --milestone 40 --mask_path D:/path/to/some_mask.png
    python scripts/visualize_denoising.py --milestone 40 --make_gif --n_frames 60
"""
import glob
import math
import os
import random
import sys
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import utils as tvutils

# Allow `python scripts/visualize_denoising.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm import GaussianDiffusion, Trainer, Unet


def _build_trainer(
    results_folder: str,
    milestone: int,
    dim: int,
    num_classes: int,
    dim_mults: list,
    channels: int,
    block_per_layer: int,
    resnet_block_groups: int,
    image_size: int,
    timesteps: int,
    sampling_timesteps: int,
):
    unet = Unet(
        dim=dim,
        num_classes=num_classes,
        dim_mults=dim_mults,
        channels=channels,
        resnet_block_groups=resnet_block_groups,
        block_per_layer=block_per_layer,
    )
    model = GaussianDiffusion(
        unet,
        image_size=image_size // 8,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps,
        loss_type='l2',
    )
    # No data folders passed — Trainer skips dataloader setup
    trainer = Trainer(
        model,
        train_folder=None,
        test_folder=None,
        train_batch_size=2,
        train_lr=1e-4,
        train_num_steps=1,
        save_and_sample_every=99999,
        num_workers=0,
        results_folder=results_folder,
    )
    trainer.load(milestone)
    trainer.ema = trainer.accelerator.unwrap_model(trainer.ema)
    trainer.ema.ema_model.eval()
    return trainer


def _pick_snapshot_step_indices(n_steps: int, n_snapshots: int, time_pairs: list,
                                target_t_values: list = None,
                                include_initial_noise: bool = False) -> list:
    """Return exactly the step indices at which to snapshot. No auto-additions.

    Special index -1 means "before any DDIM step" (the pure Gaussian noise state).

    If target_t_values is given: each target t is mapped to the nearest actual
    DDIM step; targets >= 1000 map to -1 (pure noise). n_snapshots is ignored.

    Otherwise: n_snapshots evenly-spaced step indices between 0 and n_steps-1
    (inclusive). If include_initial_noise=True, the first slot is -1 (pure
    noise) and the remaining n_snapshots-1 are evenly spaced across DDIM steps.
    """
    if target_t_values is not None:
        times_at_step = [pair[0] for pair in time_pairs]
        indices = set()
        for target in target_t_values:
            if target >= 1000:
                indices.add(-1)
            else:
                diffs = [abs(t - target) for t in times_at_step]
                indices.add(int(np.argmin(diffs)))
        return sorted(indices)

    if include_initial_noise and n_snapshots >= 2:
        interior = np.linspace(0, n_steps - 1, n_snapshots - 1).astype(int).tolist()
        return sorted(set([-1] + interior))

    return sorted(set(np.linspace(0, n_steps - 1, n_snapshots).astype(int).tolist()))


@torch.no_grad()
def _denoise_with_snapshots(
    trainer: Trainer,
    mask: torch.Tensor,
    sampling_steps: int,
    cond_scale: float,
    n_snapshots: int,
    mode: str,           # 'pred_x0', 'raw_xt', or 'both'
    device: str,
    target_t_values: list = None,
    include_initial_noise: bool = False,
):
    """Run DDIM denoising and return decoded snapshots.

    Uses a custom DDIM loop (not ddim_multimask) so we can extract the
    model's predicted x_0 at each step. For a single-class mask this is
    equivalent to what ddim_multimask does; for a multi-class mask, this
    is standard latent-diffusion DDIM (skips the DiffInfinite per-class
    mask blending that ddim_multimask does — that's a sampling-time
    quality optimization, not a training-time invariant, and it's fine to
    skip for visualization).

    Returns a list of dicts with keys:
        {'t': int, 'pred_x0_img': tensor or None, 'raw_xt_img': tensor or None}
    Sorted in DESCENDING order of t (pure noise on the left, clean on the right).
    """
    assert mode in ('pred_x0', 'raw_xt', 'both'), f'unknown mode: {mode}'

    ema_model = trainer.ema.ema_model
    vae = trainer._get_vae()

    # Build the DDIM (time, time_next) pairs the same way ddim_sample does
    times = torch.linspace(-1, ema_model.num_timesteps - 1, steps=sampling_steps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))
    n_steps = len(time_pairs)

    # Downsample the mask to latent resolution (same as ddim_sample does)
    masks = mask.clone().float()
    vmin, vmax = masks.min(), masks.max()
    masks = F.interpolate(masks, size=ema_model.image_size)
    masks = torch.clamp(torch.round(masks), vmin, vmax).int()

    # Pure-noise initial latent
    shape = (mask.shape[0], ema_model.channels, ema_model.image_size, ema_model.image_size)
    x_t = torch.randn(shape, device=device)

    # Pick snapshot step indices. Exactly what the user asked for, no
    # silent additions.
    step_indices = _pick_snapshot_step_indices(
        n_steps=n_steps,
        n_snapshots=n_snapshots,
        time_pairs=time_pairs,
        target_t_values=target_t_values,
        include_initial_noise=include_initial_noise,
    )
    step_index_set = set(step_indices)

    def _decode(latent):
        z = latent.clone() * 50
        return torch.clip(vae.decode(z).sample, 0, 1)[0].cpu()

    # Collect (t, pred_x0_img, raw_xt_img) tuples
    snapshots = []

    def _record(t_val, pred_x0_lat, x_t_lat):
        pred_x0_img = _decode(pred_x0_lat) if (mode in ('pred_x0', 'both') and pred_x0_lat is not None) else None
        raw_xt_img  = _decode(x_t_lat)     if (mode in ('raw_xt', 'both')  and x_t_lat is not None)     else None
        snapshots.append({'t': int(t_val), 'pred_x0_img': pred_x0_img, 'raw_xt_img': raw_xt_img})

    # Snapshot the initial pure-noise state.
    # For pred_x0 at step_idx=-1, we have no prediction yet — use x_t (all noise)
    # as a "prediction" (which is what it effectively is: the network's guess
    # from pure noise). Displaying x_t here matches convention.
    if -1 in step_index_set:
        _record(ema_model.num_timesteps, x_t.clone(), x_t.clone())

    # DDIM loop
    for step_idx, (time, time_next) in enumerate(time_pairs):
        time_cond = torch.full((shape[0],), time, device=device, dtype=torch.long)
        pred_noise, x_start = ema_model.model_predictions(
            x_t, time_cond, masks, cond_scale=cond_scale, clip_x_start=True,
        )

        # Snapshot BEFORE the update, because pred_x0 is what the model
        # thinks the final image is at this timestep.
        if step_idx in step_index_set:
            _record(time, x_start.clone(), x_t.clone())

        # DDIM update
        if time_next < 0:
            x_t = x_start
        else:
            alpha = ema_model.alphas_cumprod[time]
            alpha_next = ema_model.alphas_cumprod[time_next]
            sigma = ema_model.ddim_sampling_eta * (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            ).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x_t)
            x_t = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

    # Dedupe by t (keep first) and sort descending (noise → clean)
    seen = set()
    out = []
    for snap in snapshots:
        if snap['t'] not in seen:
            seen.add(snap['t'])
            out.append(snap)
    out.sort(key=lambda s: s['t'], reverse=True)
    return out


def main(
    results_folder: str = './results/bark_1024_aug_200k',
    milestone: int = 40,
    output_dir: str = './results/bark_1024_aug_200k/figs',
    # Mask source
    mask_path: str = None,
    test_folder: str = './data/diffinfinite-bark_1024',
    seed: int = 0,
    # Sampling
    sampling_steps: int = 250,
    cond_scale: float = 3.0,
    # Visualization — the ONLY knob for how many frames. No hidden overrides.
    n_snapshots: int = 6,                # exactly N frames output
    mode: str = 'pred_x0',               # 'pred_x0' | 'raw_xt' | 'both'
    include_initial_noise: bool = False, # if True, one of the N frames is t=1000 (pure noise)
    snapshot_t_values: str = None,       # optional comma-separated t list; each snapped to nearest DDIM step. Overrides n_snapshots.
    # Optional combined outputs (default OFF — individual PNG frames are always saved)
    save_grid: bool = False,             # torchvision grid PNG
    save_annotated: bool = False,        # matplotlib figure with t labels
    make_gif: bool = False,              # animated GIF from the pred_x0 row
    gif_duration_ms: int = 200,          # ms per frame in the GIF
    # Model config (must match training)
    dim: int = 256,
    num_classes: int = 4,
    dim_mults: str = '1,2,4',
    channels: int = 4,
    resnet_block_groups: int = 2,
    block_per_layer: int = 2,
    image_size: int = 512,
    timesteps: int = 1000,
    device: str = 'cuda:0',
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dim_mults_list = [int(x) for x in dim_mults.split(',')]

    # Optional user-specified target t values; else evenly-spaced step indices
    target_t_list = None
    if snapshot_t_values is not None:
        target_t_list = [int(x) for x in snapshot_t_values.split(',')]

    print(f'Loading checkpoint {milestone} from {results_folder}...')
    trainer = _build_trainer(
        results_folder=results_folder,
        milestone=milestone,
        dim=dim,
        num_classes=num_classes,
        dim_mults=dim_mults_list,
        channels=channels,
        block_per_layer=block_per_layer,
        resnet_block_groups=resnet_block_groups,
        image_size=image_size,
        timesteps=timesteps,
        sampling_timesteps=sampling_steps,
    )
    trainer.ema.to(device)

    # Pick a mask
    if mask_path:
        mask_stem = Path(mask_path).stem
        mask_arr = np.array(Image.open(mask_path))
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
    else:
        rng = random.Random(seed)
        candidates = sorted(glob.glob(os.path.join(test_folder, '*_mask.png')))
        assert candidates, f'no *_mask.png in {test_folder}'
        chosen = rng.choice(candidates)
        print(f'Using mask: {chosen}')
        mask_stem = Path(chosen).stem
        mask_arr = np.array(Image.open(chosen))

    # Center-crop to image_size (for 1024 sources)
    h, w = mask_arr.shape[:2]
    if h > image_size or w > image_size:
        top = (h - image_size) // 2
        left = (w - image_size) // 2
        mask_arr = mask_arr[top:top + image_size, left:left + image_size]

    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0).float().to(device)
    print(f'Mask shape: {tuple(mask_tensor.shape)}, classes: {np.unique(mask_arr).tolist()}')

    # Generate trajectory
    print(f'Generating snapshots (mode={mode}, n_snapshots={n_snapshots}, '
          f'include_initial_noise={include_initial_noise}) over {sampling_steps} DDIM steps...')
    torch.manual_seed(seed)
    snapshots = _denoise_with_snapshots(
        trainer=trainer,
        mask=mask_tensor,
        sampling_steps=sampling_steps,
        cond_scale=cond_scale,
        n_snapshots=n_snapshots,
        mode=mode,
        device=device,
        target_t_values=target_t_list,
        include_initial_noise=include_initial_noise,
    )

    print(f'Got {len(snapshots)} snapshots at t = {[s["t"] for s in snapshots]}')

    # ----- Primary output: individual PNGs in a per-run folder -----
    # This is what you actually asked for. Combine them yourself into figures.
    frames_dir = out / f'trajectory_m{milestone}_{mask_stem}'
    frames_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0
    for i, snap in enumerate(snapshots):
        for kind_key, kind_name in (('pred_x0_img', 'pred_x0'), ('raw_xt_img', 'raw_xt')):
            im = snap[kind_key]
            if im is None:
                continue
            fname = f'frame_{i:02d}_t{snap["t"]:04d}_{kind_name}.png'
            tvutils.save_image(im, str(frames_dir / fname))
            n_saved += 1
    print(f'Saved {n_saved} individual PNG frames to {frames_dir}')

    # ----- Opt-in combined outputs -----
    rows = []
    if mode in ('pred_x0', 'both'):
        rows.append(('pred_x0', [s['pred_x0_img'] for s in snapshots]))
    if mode in ('raw_xt', 'both'):
        rows.append(('raw_xt', [s['raw_xt_img'] for s in snapshots]))

    if save_grid:
        for row_name, row_imgs in rows:
            row_imgs_valid = [im for im in row_imgs if im is not None]
            if not row_imgs_valid:
                continue
            imgs_t = torch.stack(row_imgs_valid)
            grid = tvutils.make_grid(imgs_t, nrow=len(row_imgs_valid), padding=4, pad_value=1.0)
            grid_path = out / f'grid_m{milestone}_{mask_stem}_{row_name}.png'
            tvutils.save_image(grid, str(grid_path))
            print(f'Saved grid: {grid_path}')

    if save_annotated:
        try:
            import matplotlib.pyplot as plt
            n_col = len(snapshots)
            n_row = len(rows)
            _, axes = plt.subplots(n_row, n_col, figsize=(2.2 * n_col, 2.7 * n_row),
                                   squeeze=False)
            for r, (row_name, row_imgs) in enumerate(rows):
                for c, (snap, im) in enumerate(zip(snapshots, row_imgs)):
                    ax = axes[r, c]
                    if im is None:
                        ax.axis('off')
                        continue
                    ax.imshow(im.permute(1, 2, 0).numpy())
                    if r == 0:
                        ax.set_title(f't = {snap["t"]}', fontsize=10)
                    if c == 0:
                        ax.set_ylabel(row_name, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])
            plt.tight_layout()
            ann_path = out / f'annotated_m{milestone}_{mask_stem}.png'
            plt.savefig(ann_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'Saved annotated: {ann_path}')
        except ImportError:
            print('matplotlib not installed; skipping annotated figure')

    if make_gif:
        try:
            # Prefer pred_x0 for the GIF; fall back to first available row
            gif_row = None
            for row_name, row_imgs in rows:
                if row_name == 'pred_x0':
                    gif_row = row_imgs
                    break
            if gif_row is None and rows:
                gif_row = rows[0][1]
            frames = []
            for im in gif_row:
                if im is None:
                    continue
                arr = (im.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                frames.append(Image.fromarray(arr))
            if frames:
                gif_path = out / f'trajectory_m{milestone}_{mask_stem}.gif'
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=gif_duration_ms,
                    loop=0,
                )
                print(f'Saved GIF: {gif_path}')
        except Exception as e:
            print(f'GIF generation failed: {e}')

    print('Done.')


if __name__ == '__main__':
    fire.Fire(main)
