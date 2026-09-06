"""FID + KID as a function of DDIM sampling steps, at a fixed checkpoint.

The quality-vs-compute trade-off figure for the thesis: how many denoising
steps does the sampler actually need? Sweeps a range of DDIM step counts
(default 10, 25, 50, 100, 250, 500, 1000) on one checkpoint (default: the
highest-numbered model-N.pt), generating K samples per center-cropped source
mask for each setting and computing FID + KID against the same center-cropped
real reference used by eval_checkpoints.py — so numbers are directly
comparable between the two experiments.

All results are appended to <results_folder>/sampling_steps_sweep.csv, so the
plot can always be regenerated from data alone:

    python scripts/eval_sampling_steps.py --plot_only

Plots are written as PDF (vector, lossless — include directly in LaTeX via
\\includegraphics) plus a PNG preview. All plot text is in Slovene.

CSV columns:
    sampling_steps, milestone, step, fid, kid_mean, kid_std, n_gen, gen_seconds

`gen_seconds` (wall-clock generation time for the whole set) is recorded so a
quality-vs-compute plot can be made later from the same CSV.

Implementation note: the per-call `sampling_timesteps` argument of
GaussianDiffusion.sample() overrides the constructor value, so the model is
loaded ONCE and reused for every step-count setting. The constructor uses a
dummy value of 250 purely to keep `is_ddim_sampling=True`; the 1000-step
setting still runs through the DDIM path (linspace over all 1000 timesteps),
which is what we want for a controlled comparison.

Usage:
    python scripts/eval_sampling_steps.py                       # full sweep
    python scripts/eval_sampling_steps.py --steps_list 50,250   # subset
    python scripts/eval_sampling_steps.py --plot_only           # replot from CSV
    python scripts/eval_sampling_steps.py --device cuda:1 --fid_gpu 1

Estimated compute (RTX 4000 Ada, 269 masks, K=4 → 1076 generations/setting):
    time scales linearly with steps; ~18 min at 50 steps, ~6 h at 1000 steps.
    Full default sweep (10+25+50+100+250+500+1000 = 1935 step-units)
    ≈ 1935/50 × 18 min ≈ 11.5 hours. Plan as an overnight run.

Requires: pip install torch-fidelity
"""
import csv
import shutil
import sys
import time
from pathlib import Path

import fire
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))         # scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from eval_checkpoints import (
    _build_trainer,
    _compute_metrics,
    _discover_milestones,
    _generate_for_checkpoint,
    _prepare_eval_data,
)


def _plot(csv_path: Path, results_folder: Path, out_stem: str = 'sampling_steps_sweep'):
    """Render the FID/KID-vs-steps figure from the CSV. Slovene labels,
    PDF (vector) + PNG preview."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    'sampling_steps': int(row['sampling_steps']),
                    'fid':            float(row['fid']),
                    'kid_mean':       float(row['kid_mean']),
                    'kid_std':        float(row['kid_std']),
                })
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r['sampling_steps'])

    if not rows:
        print('V CSV ni dokončanih vrstic; ni kaj narisati.')
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # LaTeX-friendly sizing: readable at \textwidth in a two-panel layout.
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 10,
        'pdf.fonttype': 42,   # embed TrueType — text stays selectable/editable
    })

    steps    = [r['sampling_steps'] for r in rows]
    fids     = [r['fid']            for r in rows]
    kid_means = [r['kid_mean']      for r in rows]
    kid_stds  = [r['kid_std']       for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    # FID panel
    ax = axes[0]
    ax.plot(steps, fids, marker='o', linewidth=1.7, color='C0')
    best = int(np.argmin(fids))
    ax.axvline(steps[best], color='red', alpha=0.4, linestyle='--',
               label=f'najmanjši FID = {fids[best]:.2f}\npri {steps[best]} korakih')
    ax.set_xscale('log')
    ax.set_xticks(steps)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel('Število korakov vzorčenja DDIM')
    ax.set_ylabel('FID')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    # KID panel
    ax = axes[1]
    ax.errorbar(steps, kid_means, yerr=kid_stds, marker='o', linewidth=1.7,
                color='C1', ecolor='C1', capsize=3, alpha=0.9)
    best = int(np.argmin(kid_means))
    ax.axvline(steps[best], color='red', alpha=0.4, linestyle='--',
               label=f'najmanjši KID = {kid_means[best]:.4f}\npri {steps[best]} korakih')
    ax.set_xscale('log')
    ax.set_xticks(steps)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel('Število korakov vzorčenja DDIM')
    ax.set_ylabel('KID (povprečje ± std podmnožic)')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    fig.suptitle('Vpliv števila korakov vzorčenja na kakovost generiranih slik')
    plt.tight_layout()

    out_pdf = results_folder / f'{out_stem}.pdf'
    out_png = results_folder / f'{out_stem}.png'
    plt.savefig(out_pdf)             # vector — for LaTeX
    plt.savefig(out_png, dpi=200)    # raster preview
    plt.close()
    print(f'Shranjeno: {out_pdf}')
    print(f'Shranjeno: {out_png}')


def main(
    results_folder: str = './results/bark_1024_aug_200k',
    source_folder: str = './data/diffinfinite-bark_1024',
    output_root: str = None,
    milestone: int = None,           # default: highest model-N.pt found
    steps_list: str = '10,25,50,100,250,500,1000',
    K: int = 4,
    cond_scale: float = 3.0,
    fid_gpu: int = 0,
    skip_existing: bool = True,
    plot_only: bool = False,
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
    csv_path = results_folder / 'sampling_steps_sweep.csv'

    if plot_only:
        _plot(csv_path, results_folder)
        return

    dim_mults_list = [int(x) for x in dim_mults.split(',')]
    steps_values = sorted({int(s) for s in steps_list.split(',')})
    assert all(1 <= s <= timesteps for s in steps_values), \
        f'sampling steps must be in [1, {timesteps}]'

    if milestone is None:
        found = _discover_milestones(results_folder)
        assert found, f'No model-*.pt in {results_folder}'
        milestone = found[-1]
    print(f'Checkpoint: model-{milestone}.pt   sweep: {steps_values}')

    # Deterministic eval reference (shared with eval_checkpoints.py)
    real_images_dir, masks_dir = _prepare_eval_data(
        source_folder, output_root / 'eval_data', crop_size=image_size,
    )
    eval_masks = sorted(masks_dir.glob('*_mask.png'))
    n_real = len(list(real_images_dir.glob('*.jpg')))
    n_gen = len(eval_masks) * K
    print(f'Eval reference: {n_real} reals; {len(eval_masks)} masks × K={K} = {n_gen} generations per setting')

    if kid_subset_size is None:
        kid_subset_size = min(n_real, n_gen)
    print(f'KID subset size: {kid_subset_size}')

    # Resume support
    already_done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    float(row['fid']); float(row['kid_mean'])
                    already_done.add(int(row['sampling_steps']))
                except (KeyError, ValueError):
                    continue
        print(f'Resuming: {len(already_done)} settings already in {csv_path.name}')
    else:
        with open(csv_path, 'w') as f:
            f.write('sampling_steps,milestone,step,fid,kid_mean,kid_std,n_gen,gen_seconds\n')

    # Load the model ONCE. Constructor sampling_steps=250 is a dummy that
    # keeps is_ddim_sampling=True; each generation call overrides it.
    print(f'Loading model-{milestone}.pt ...')
    trainer = _build_trainer(
        str(results_folder), milestone,
        dim, num_classes, dim_mults_list, channels,
        block_per_layer, resnet_block_groups,
        image_size, timesteps, 250,
    )
    trainer.ema.to(device)
    step = trainer.step

    sweep_root = output_root / f'steps_sweep_cp{milestone:03d}'

    for s in steps_values:
        if skip_existing and s in already_done:
            print(f'[skip] {s} korakov: že v CSV')
            continue

        gen_dir = sweep_root / f'steps_{s:04d}' / 'gen'
        t0 = time.time()
        print(f'\n=== {s} korakov vzorčenja ===')

        gen_cached = (
            gen_dir.exists()
            and skip_existing
            and len(list(gen_dir.glob('*.png'))) >= n_gen
        )
        if gen_cached:
            print(f'  cached ({len(list(gen_dir.glob("*.png")))} PNGs), skipping sampling')
            gen_seconds = float('nan')
        else:
            if gen_dir.exists():
                shutil.rmtree(gen_dir)
            print(f'  generating {n_gen} samples (sampling_steps={s}, cond_scale={cond_scale})...')
            n = _generate_for_checkpoint(
                trainer, eval_masks, K, s, cond_scale, gen_dir, device,
            )
            gen_seconds = time.time() - t0
            print(f'  saved {n} PNGs in {gen_seconds/60:.1f} min')

        print(f'  computing FID + KID ...')
        try:
            metrics = _compute_metrics(
                gen_dir, real_images_dir,
                kid_subset_size=kid_subset_size, gpu_id=fid_gpu,
            )
        except Exception as e:
            print(f'  metrics failed: {e}')
            continue

        print(f'  FID = {metrics["fid"]:.3f}   '
              f'KID = {metrics["kid_mean"]:.4f} ± {metrics["kid_std"]:.4f}   '
              f'(total {(time.time()-t0)/60:.1f} min)')

        with open(csv_path, 'a') as f:
            f.write(
                f'{s},{milestone},{step},'
                f'{metrics["fid"]},{metrics["kid_mean"]},{metrics["kid_std"]},'
                f'{n_gen},{gen_seconds}\n'
            )

    del trainer
    torch.cuda.empty_cache()

    _plot(csv_path, results_folder)
    print('\nDone.')


if __name__ == '__main__':
    fire.Fire(main)
