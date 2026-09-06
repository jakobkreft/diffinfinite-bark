"""FID + KID curves across saved checkpoints — single-curve, in-distribution.

The training pipeline now uses random 512x512 crops from 1024x1024 sources
with no held-out test set, so the eval is also in-distribution: we sample
the model conditioned on the deterministic center crop of each source mask,
and compare against the deterministic center crop of each source image.

For each `model-N.pt` in `--results_folder`:

    1. Load the checkpoint (EMA weights).
    2. Generate K samples per (center-cropped) mask → cp_NNN/gen/ (N_src × K).
    3. Compute FID + KID against the center-cropped real reference.
    4. Append a row to <results_folder>/fid_curve.csv.

After the sweep, plot FID and KID as two single-line panels to
<results_folder>/fid_curve.png.

Why center crops, not random crops:
- The eval must be deterministic across checkpoints. If the masks (and
  reals) varied per checkpoint, the FID curve would be confounded.
- The model saw many random crops of each source during training, so the
  specific center crop the eval uses was almost certainly seen too. This
  is fundamentally an in-distribution measure, not a generalization
  measure. We frame it that way in the thesis: tracks "did training
  improve the match to the data distribution?" rather than "does it
  generalize to unseen patches?".

Why FID + KID together:
- FID has positive bias at small reference sizes. With 270 source images
  cropped to 270 reals, FID will carry visible bias.
- KID's U-statistic estimator is approximately unbiased at small N. Use
  KID as the headline metric in the thesis; FID for cross-paper context.

Disk layout under output_root:
    eval_data/images/                   (center-cropped 512x512 reals)
    eval_data/masks/                    (center-cropped 512x512 masks)
    cp_{milestone:03d}/gen/             (K × N_src generated PNGs)

Usage (typical):
    python scripts/eval_checkpoints.py \\
        --results_folder ./results/bark_1024_aug_200k \\
        --source_folder ./data/diffinfinite-bark_1024 \\
        --K 4 --sampling_steps 50

    # Specific milestones only:
    python scripts/eval_checkpoints.py --milestones 5,10,15

    # Sample on cuda:1 while training runs on cuda:0:
    python scripts/eval_checkpoints.py --device cuda:1 --fid_gpu 1

Estimated compute (RTX 4000 Ada, 512x512, sampling_steps=50, K=4, 270 sources):
    270 × 4 = 1080 generations per checkpoint
    ≈ 18 min per checkpoint  →  40 checkpoints ≈ 12 hours

Requires: pip install torch-fidelity
"""
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import fire
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm import GaussianDiffusion, Trainer, Unet


def _build_trainer(results_folder, milestone, dim, num_classes, dim_mults,
                   channels, block_per_layer, resnet_block_groups,
                   image_size, timesteps, sampling_steps):
    unet = Unet(
        dim=dim, num_classes=num_classes, dim_mults=dim_mults,
        channels=channels, resnet_block_groups=resnet_block_groups,
        block_per_layer=block_per_layer,
    )
    model = GaussianDiffusion(
        unet, image_size=image_size // 8, timesteps=timesteps,
        sampling_timesteps=sampling_steps, loss_type='l2',
    )
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


def _save_uint8(tensor, path):
    arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, format='PNG', optimize=False)


@torch.no_grad()
def _generate_for_checkpoint(trainer, mask_files, K, sampling_steps, cond_scale, out_dir, device):
    """Generate K samples per mask, save as PNGs in out_dir. Returns count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vae = trainer._get_vae()
    ema_model = trainer.ema.ema_model

    n_saved = 0
    for mask_path in mask_files:
        stem = Path(mask_path).stem.replace('_mask', '')
        mask_arr = np.array(Image.open(mask_path))
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0).float().to(device)
        # Replicate K times to do K samples in batches; respect VRAM.
        BATCH = min(K, 4)  # 512x512, 4 simultaneous samples is OK on 20GB
        for batch_start in range(0, K, BATCH):
            batch_n = min(BATCH, K - batch_start)
            m_batch = mask_tensor.repeat(batch_n, 1, 1, 1)
            z = torch.randn(batch_n, ema_model.channels, ema_model.image_size, ema_model.image_size, device=device)
            z = ema_model.sample(z, m_batch, sampling_timesteps=sampling_steps, cond_scale=cond_scale) * 50
            imgs = torch.clip(vae.decode(z).sample, 0, 1)
            for i in range(batch_n):
                _save_uint8(imgs[i], out_dir / f'{stem}_k{batch_start + i:03d}.png')
                n_saved += 1
    return n_saved


def _compute_metrics(generated_dir: Path, test_images_dir: Path,
                     kid_subset_size: int, gpu_id: int = 0) -> dict:
    """Call torch-fidelity CLI for both FID and KID in one invocation.

    KID is the more reliable metric on small real-set sizes; FID has a known
    positive bias when one side has few hundred samples (which we do — 39
    real test images). Reporting both lets reviewers see the agreement.

    Returns {'fid': float, 'kid_mean': float, 'kid_std': float}.
    """
    cmd = [
        'fidelity', '--gpu', str(gpu_id), '--fid', '--kid',
        '--kid-subset-size', str(kid_subset_size),
        '--input1', str(generated_dir),
        '--input2', str(test_images_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print('torch-fidelity stderr:', result.stderr)
        raise RuntimeError(f'fidelity exited {result.returncode}')

    out = {}
    patterns = {
        'fid':      r'\s*frechet_inception_distance:\s*([\d.eE+-]+)',
        'kid_mean': r'\s*kernel_inception_distance_mean:\s*([\d.eE+-]+)',
        'kid_std':  r'\s*kernel_inception_distance_std:\s*([\d.eE+-]+)',
    }
    for line in result.stdout.splitlines():
        for key, pat in patterns.items():
            m = re.match(pat, line)
            if m:
                out[key] = float(m.group(1))

    missing = set(patterns) - set(out)
    if missing:
        raise RuntimeError(
            f'Missing metrics in fidelity output: {missing}\nstdout={result.stdout!r}'
        )
    return out


def _discover_milestones(results_folder: Path) -> list:
    pts = sorted(
        results_folder.glob('model-*.pt'),
        key=lambda p: int(re.search(r'model-(\d+)\.pt', p.name).group(1)),
    )
    return [int(re.search(r'model-(\d+)\.pt', p.name).group(1)) for p in pts]


def _prepare_eval_data(source_folder: Path, target_root: Path, crop_size: int = 512) -> tuple:
    """Build the deterministic eval reference and the deterministic mask set.

    For each `<stem>.jpg` + `<stem>_mask.png` in `source_folder`:
      - Center-crop the image to `crop_size` x `crop_size`, save to
        `target_root/images/<stem>.jpg`.
      - Center-crop the mask the same way (NEAREST-style — PIL's `crop` is
        coordinate-based, no interpolation), save to
        `target_root/masks/<stem>_mask.png`.

    If the source is smaller than `crop_size`, we error out. If equal, the
    crop is a no-op (passes through).

    Idempotent: if both target dirs are already fully populated, no work.

    Returns (images_dir, masks_dir).
    """
    images_dir = target_root / 'images'
    masks_dir  = target_root / 'masks'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    src_jpgs = sorted(source_folder.glob('*.jpg'))
    assert src_jpgs, f'no *.jpg in {source_folder}'

    n_existing_img  = len(list(images_dir.glob('*.jpg')))
    n_existing_mask = len(list(masks_dir.glob('*_mask.png')))
    if n_existing_img >= len(src_jpgs) and n_existing_mask >= len(src_jpgs):
        return images_dir, masks_dir

    for jpg in src_jpgs:
        stem = jpg.stem
        src_mask = source_folder / f'{stem}_mask.png'
        if not src_mask.exists():
            print(f'  WARN: missing mask for {stem}, skipping')
            continue

        dst_jpg  = images_dir / jpg.name
        dst_mask = masks_dir / src_mask.name

        # Image: center crop with PIL (Lanczos-quality not needed; we crop)
        if not dst_jpg.exists():
            im = Image.open(jpg).convert('RGB')
            assert im.size[0] >= crop_size and im.size[1] >= crop_size, \
                f'{jpg.name} is {im.size}, smaller than crop_size={crop_size}'
            w, h = im.size
            left = (w - crop_size) // 2
            top  = (h - crop_size) // 2
            im_cropped = im.crop((left, top, left + crop_size, top + crop_size))
            im_cropped.save(dst_jpg, format='JPEG', quality=95)

        # Mask: same crop coordinates, NEAREST preserves class integer values
        if not dst_mask.exists():
            mk = Image.open(src_mask)
            w, h = mk.size
            left = (w - crop_size) // 2
            top  = (h - crop_size) // 2
            mk_cropped = mk.crop((left, top, left + crop_size, top + crop_size))
            mk_cropped.save(dst_mask, format='PNG', optimize=False)

    return images_dir, masks_dir


def main(
    results_folder: str = './results/bark_1024_aug_200k',
    source_folder: str = './data/diffinfinite-bark_1024',
    output_root: str = None,
    milestones: str = None,
    K: int = 4,
    sampling_steps: int = 50,
    cond_scale: float = 3.0,
    fid_gpu: int = 0,
    skip_existing: bool = True,
    # Defaults to min(n_real, n_gen). User can override.
    kid_subset_size: int = None,
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
    results_folder = Path(results_folder)
    source_folder = Path(source_folder)
    output_root = Path(output_root) if output_root else (results_folder / 'eval')
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_folder / 'fid_curve.csv'

    dim_mults_list = [int(x) for x in dim_mults.split(',')]

    # Discover milestones
    if milestones is None:
        target_milestones = _discover_milestones(results_folder)
        if not target_milestones:
            print(f'No model-*.pt files in {results_folder}.')
            return
    else:
        target_milestones = [int(x) for x in milestones.split(',')]

    print(f'Will evaluate {len(target_milestones)} checkpoints: {target_milestones}')

    # Build the deterministic eval reference (center-cropped to 512)
    real_images_dir, masks_dir = _prepare_eval_data(
        source_folder, output_root / 'eval_data', crop_size=image_size,
    )
    eval_masks = sorted(masks_dir.glob('*_mask.png'))
    n_real = len(list(real_images_dir.glob('*.jpg')))
    print(f'Eval reference: {n_real} center-cropped real images at {real_images_dir}')
    print(f'Eval masks:     {len(eval_masks)} center-cropped masks at {masks_dir}')

    n_gen = len(eval_masks) * K
    print(f'Per-checkpoint sampling budget: {n_gen} generations '
          f'({len(eval_masks)} masks × K={K})')

    # KID subset size — torch-fidelity caps at min(real, fake).
    if kid_subset_size is None:
        kid_subset_size = min(n_real, n_gen)
    print(f'KID subset size: {kid_subset_size}')

    # Resume support: skip a milestone if it already has a complete row.
    already_done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    for k in ('fid', 'kid_mean'):
                        float(row[k])
                    already_done.add(int(row['milestone']))
                except (KeyError, ValueError):
                    continue
        print(f'Resuming: {len(already_done)} milestones already complete in fid_curve.csv')
    else:
        with open(csv_path, 'w') as f:
            f.write('milestone,step,fid,kid_mean,kid_std,n_gen\n')

    for milestone in target_milestones:
        if skip_existing and milestone in already_done:
            print(f'[skip] milestone {milestone}: already in CSV')
            continue

        cp_root = output_root / f'cp_{milestone:03d}'
        gen_dir = cp_root / 'gen'
        t0 = time.time()

        print(f'\n=== Milestone {milestone} ===')

        # Decide whether sampling is needed
        gen_cached = (
            gen_dir.exists()
            and skip_existing
            and len(list(gen_dir.glob('*.png'))) >= n_gen
        )

        if not gen_cached:
            if gen_dir.exists():
                shutil.rmtree(gen_dir)
            print(f'  loading model-{milestone}.pt for sampling...')
            trainer = _build_trainer(
                str(results_folder), milestone,
                dim, num_classes, dim_mults_list, channels,
                block_per_layer, resnet_block_groups,
                image_size, timesteps, sampling_steps,
            )
            trainer.ema.to(device)
            step = trainer.step
            print(f'  generating {n_gen} samples '
                  f'(sampling_steps={sampling_steps}, cond_scale={cond_scale})...')
            n = _generate_for_checkpoint(
                trainer, eval_masks, K, sampling_steps, cond_scale, gen_dir, device,
            )
            print(f'  saved {n} PNGs to {gen_dir}')
            del trainer
            torch.cuda.empty_cache()
        else:
            # Cached — load just the step number.
            ckpt = torch.load(
                str(results_folder / f'model-{milestone}.pt'),
                map_location='cpu', weights_only=False,
            )
            step = int(ckpt.get('step', milestone * 1000))
            del ckpt
            print(f'  cached, skipping sampling ({n_gen} PNGs present)')

        # Compute FID + KID against the shared center-cropped reference
        print(f'  computing FID + KID ...')
        try:
            metrics = _compute_metrics(
                gen_dir, real_images_dir,
                kid_subset_size=kid_subset_size, gpu_id=fid_gpu,
            )
        except Exception as e:
            print(f'  metrics failed: {e}')
            continue

        elapsed = time.time() - t0
        print(f'  FID = {metrics["fid"]:.3f}   '
              f'KID = {metrics["kid_mean"]:.4f} ± {metrics["kid_std"]:.4f}   '
              f'(elapsed {elapsed/60:.1f} min)')

        with open(csv_path, 'a') as f:
            f.write(
                f'{milestone},{step},'
                f'{metrics["fid"]},{metrics["kid_mean"]},{metrics["kid_std"]},'
                f'{n_gen}\n'
            )

    # Plot — single FID curve + single KID curve. Reads the full csv so
    # partial sweeps still produce a graph.
    print(f'\nReading {csv_path} to plot...')
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    'milestone': int(row['milestone']),
                    'step':      int(row['step']),
                    'fid':       float(row['fid']),
                    'kid_mean':  float(row['kid_mean']),
                    'kid_std':   float(row['kid_std']),
                })
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r['step'])

    if not rows:
        print('No completed rows in CSV; nothing to plot.')
        print('\nDone.')
        return

    try:
        import matplotlib.pyplot as plt
        steps    = [r['step']     for r in rows]
        fids     = [r['fid']      for r in rows]
        kid_means = [r['kid_mean'] for r in rows]
        kid_stds  = [r['kid_std']  for r in rows]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

        # FID panel
        ax = axes[0]
        ax.plot(steps, fids, marker='o', linewidth=1.7, color='C0')
        best_idx = int(np.argmin(fids))
        ax.axvline(steps[best_idx], color='red', alpha=0.4, linestyle='--',
                   label=f'best: step {steps[best_idx]}, FID {fids[best_idx]:.2f}')
        ax.set_xlabel('Training step')
        ax.set_ylabel('FID (vs center-cropped reals)')
        ax.set_title('FID vs training step')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # KID panel (error bars = subset std)
        ax = axes[1]
        ax.errorbar(steps, kid_means, yerr=kid_stds, marker='o', linewidth=1.7,
                    color='C0', ecolor='C0', capsize=3, alpha=0.9)
        best_idx = int(np.argmin(kid_means))
        ax.axvline(steps[best_idx], color='red', alpha=0.4, linestyle='--',
                   label=f'best: step {steps[best_idx]}, KID {kid_means[best_idx]:.4f}')
        ax.set_xlabel('Training step')
        ax.set_ylabel('KID (mean ± std over subsets)')
        ax.set_title('KID vs training step')
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.suptitle(
            f'Evaluation curve — {results_folder.name}\n'
            f'In-distribution: {len(eval_masks)} center-cropped reference images, '
            f'{K} samples per mask, sampling_steps={sampling_steps}'
        )
        out_png = results_folder / 'fid_curve.png'
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f'Saved plot: {out_png}')
    except ImportError:
        print('matplotlib not installed; skipping plot')

    print('\nDone.')


if __name__ == '__main__':
    fire.Fire(main)
