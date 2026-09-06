"""GLCM statistike — kontrast, homogenost, energija, korelacija.

Lokalna teksturna statistika drugega reda: ali imajo generirane teksture
podobno "zrnatost" kot realne. Implementacija po standardnem receptu
(Haralick; skimage.feature.graycomatrix + graycoprops):

    1. Iz vsake slike vzamemo `crops_per_image` naključnih sivinskih izsekov
       velikosti `crop_size` (privzeto 256×256), determinirano s semenom.
    2. Sivine kvantiziramo na `levels` nivojev (privzeto 64).
    3. GLCM: zamiki `distances` (privzeto 1 in 5 px), štirje koti
       (0°, 45°, 90°, 135°), symmetric=True, normed=True.
    4. graycoprops za vsako od štirih lastnosti; vrednosti povprečimo čez
       kote (rotacijska invarianca), ločeno po zamiku.
    5. Poročamo povprečje ± odklon čez vse izseke, za realni in generirani
       nabor. Bližje skupaj = boljše ujemanje teksturne statistike.

Rezultati:
    glcm_raw.csv    — ena vrstica na (izsek, lastnost, zamik); iz tega se
                      graf vedno lahko obnovi (--plot_only)
    glcm_summary.csv — povprečje ± odklon po (nabor, lastnost, zamik)
    glcm_stats.pdf/.png — 2×2 mreža (ena lastnost na panel), stolpci za
                      realne/generirane z intervali ±1σ, po zamikih.
                      Vsi napisi v slovenščini.

Uporaba:
    python scripts/eval_glcm.py
    python scripts/eval_glcm.py --real_dir ... --gen_dir ...
    python scripts/eval_glcm.py --plot_only

Zahteva: scikit-image (že v requirements.txt).
"""
import csv
import random
import sys
from pathlib import Path

import fire
import numpy as np
from PIL import Image

PROPS = ['contrast', 'homogeneity', 'energy', 'correlation']
PROPS_SI = {
    'contrast': 'kontrast',
    'homogeneity': 'homogenost',
    'energy': 'energija',
    'correlation': 'korelacija',
}


def _list_images(folder: Path):
    files = sorted(list(folder.glob('*.png')) + list(folder.glob('*.jpg')))
    assert files, f'no images in {folder}'
    return files


def _glcm_props_for_crop(gray_q, distances, levels):
    """GLCM props for one quantized grayscale crop.

    Returns {(prop, distance): value} with values averaged over the four
    angles (rotation invariance).
    """
    from skimage.feature import graycomatrix, graycoprops
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_q, distances=distances, angles=angles,
                        levels=levels, symmetric=True, normed=True)
    out = {}
    for prop in PROPS:
        vals = graycoprops(glcm, prop)          # (n_distances, n_angles)
        for di, d in enumerate(distances):
            out[(prop, d)] = float(vals[di].mean())
    return out


def _collect(files, label, crops_per_image, crop_size, levels, distances, seed):
    """Rows of (set, file, crop_idx, prop, distance, value) over a file list."""
    rng = random.Random(seed)
    rows = []
    for fi, f in enumerate(files):
        img = Image.open(f).convert('L')
        w, h = img.size
        assert w >= crop_size and h >= crop_size, f'{f.name} smaller than crop_size'
        arr = np.asarray(img)
        for ci in range(crops_per_image):
            x = rng.randint(0, w - crop_size)
            y = rng.randint(0, h - crop_size)
            crop = arr[y:y + crop_size, x:x + crop_size]
            # quantize 0..255 -> 0..levels-1
            q = (crop.astype(np.uint16) * levels // 256).astype(np.uint8)
            props = _glcm_props_for_crop(q, distances, levels)
            for (prop, d), v in props.items():
                rows.append((label, f.name, ci, prop, d, v))
        print(f'\r  [{label}] {fi + 1}/{len(files)}', end='')
    print()
    return rows


def _plot(raw_csv: Path, out_dir: Path, out_stem: str = 'glcm_stats'):
    # Aggregate raw rows -> mean/std by (set, prop, distance)
    data = {}
    with open(raw_csv, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                key = (row['set'], row['prop'], int(row['distance']))
                data.setdefault(key, []).append(float(row['value']))
            except (KeyError, ValueError):
                continue
    if not data:
        print('V CSV ni vrstic; ni kaj narisati.')
        return

    sets = sorted({k[0] for k in data})
    distances = sorted({k[2] for k in data})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'legend.fontsize': 10, 'pdf.fonttype': 42,
    })

    set_style = {}   # set -> (color, slovene label)
    palette = ['C0', 'C1', 'C2', 'C3']
    si_names = {'real': 'realne', 'gen': 'generirane'}
    for i, s in enumerate(sets):
        set_style[s] = (palette[i % len(palette)], si_names.get(s, s))

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
    for ax, prop in zip(axes.flat, PROPS):
        x = np.arange(len(distances))
        n_sets = len(sets)
        w = 0.8 / n_sets
        for si, s in enumerate(sets):
            means = [np.mean(data[(s, prop, d)]) for d in distances]
            stds  = [np.std(data[(s, prop, d)])  for d in distances]
            color, label = set_style[s]
            ax.bar(x + (si - (n_sets - 1) / 2) * w, means, w,
                   yerr=stds, capsize=4, color=color, label=label, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f'zamik d = {d} px' for d in distances])
        ax.set_title(PROPS_SI[prop])
        ax.grid(True, alpha=0.3, axis='y')
    axes.flat[0].legend()
    fig.suptitle('GLCM statistike — realne proti generiranim zaplatam (povprečje ± σ)')
    plt.tight_layout()

    out_pdf = out_dir / f'{out_stem}.pdf'
    out_png = out_dir / f'{out_stem}.png'
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f'Shranjeno: {out_pdf}')
    print(f'Shranjeno: {out_png}')


def main(
    real_dir: str = './results/bark_1024_aug_200k/eval/eval_data/images',
    # 1000-step DDIM generations (best FID/KID in the sampling-steps sweep;
    # cp_040/gen was sampled at only 50 steps and is visibly worse).
    gen_dir: str = './results/bark_1024_aug_200k/eval/steps_sweep_cp040/steps_1000/gen',
    out_dir: str = './results/bark_1024_aug_200k/texture_eval',
    crops_per_image: int = 2,
    crop_size: int = 256,
    levels: int = 64,
    distances: str = '1,5',
    seed: int = 42,
    max_images: int = None,
    plot_only: bool = False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / 'glcm_raw.csv'
    summary_csv = out_dir / 'glcm_summary.csv'

    if plot_only:
        _plot(raw_csv, out_dir)
        return

    dist_list = [int(d) for d in str(distances).split(',')]

    real_files = _list_images(Path(real_dir))
    gen_files  = _list_images(Path(gen_dir))
    if max_images:
        real_files = real_files[:max_images]
        gen_files  = gen_files[:max_images]
    print(f'Realne: {len(real_files)}   Generirane: {len(gen_files)}   '
          f'izseki: {crops_per_image}×{crop_size}px, {levels} nivojev, zamiki {dist_list}')

    rows = []
    rows += _collect(real_files, 'real', crops_per_image, crop_size, levels, dist_list, seed)
    rows += _collect(gen_files,  'gen',  crops_per_image, crop_size, levels, dist_list, seed + 1)

    with open(raw_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['set', 'file', 'crop_idx', 'prop', 'distance', 'value'])
        writer.writerows(rows)
    print(f'CSV (surovi): {raw_csv}')

    # Summary table
    agg = {}
    for s, _, _, prop, d, v in rows:
        agg.setdefault((s, prop, d), []).append(v)
    with open(summary_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['set', 'prop', 'distance', 'mean', 'std', 'n_crops'])
        print(f'\n{"nabor":6s} {"lastnost":12s} {"zamik":>5s} {"povprečje":>12s} {"σ":>10s}')
        for (s, prop, d), vals in sorted(agg.items()):
            m, sd = np.mean(vals), np.std(vals)
            writer.writerow([s, prop, d, f'{m:.6f}', f'{sd:.6f}', len(vals)])
            print(f'{s:6s} {prop:12s} {d:5d} {m:12.5f} {sd:10.5f}')
    print(f'\nCSV (povzetek): {summary_csv}')

    _plot(raw_csv, out_dir)


if __name__ == '__main__':
    fire.Fire(main)
