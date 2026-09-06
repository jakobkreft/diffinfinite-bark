"""Porazdelitev velikosti povezanih komponent v semantičnih maskah.

Ali se slepice in poškodbe pojavljajo v realistično velikih in številčnih
"otokih" (ne ena velika ploskev, ne prah)?

Za vsak vir mask in vsak razred (privzeto 1 = slepice, 2 = poškodbe):

    1. binarna maska razreda -> scipy.ndimage.label (4-sosednost),
    2. velikosti komponent v slikovnih elementih (np.bincount po oznakah),
    3. dolgi CSV z eno vrstico na komponento (component_sizes.csv) — iz
       njega se graf vedno lahko obnovi (--plot_only),
    4. povzetek po (vir, razred): število komponent, komponent na sliko,
       povprečje / mediana / maksimum velikosti (component_summary.csv),
    5. histogram porazdelitve velikosti z logaritemskimi koši, normiran na
       delež komponent, prekrivno čez vire; en panel na razred.
       PDF (vektorski, za LaTeX) + PNG, vsi napisi v slovenščini.

Vire podaš kot poimenovane pare `ime=pot`, ločene z vejico. Primer za
primerjavo realnih mask s segmentiranimi generiranimi (ko bo segmentator
pognan) ali z MRF/WFC zemljevidi:

    python scripts/eval_components.py --sources ^
        "realne=./results/bark_1024_aug_200k/eval/eval_data/masks,MRF=./MRF_masks"

V mapi vira se upoštevajo datoteke *_mask.png; če jih ni, vse *.png.

Uporaba:
    python scripts/eval_components.py                # privzeto: realne maske
    python scripts/eval_components.py --plot_only
"""
import csv
import sys
from pathlib import Path

import fire
import numpy as np
from PIL import Image
from scipy import ndimage

CLASS_SI_DEFAULT = {1: 'slepice', 2: 'poškodbe'}


def _list_masks(folder: Path):
    files = sorted(folder.glob('*_mask.png'))
    if not files:
        files = sorted(folder.glob('*.png'))
    assert files, f'no mask PNGs in {folder}'
    return files


def _component_sizes(mask_arr: np.ndarray, class_value: int):
    """Sizes (px) of 4-connected components of `class_value` in the mask."""
    binary = (mask_arr == class_value)
    if not binary.any():
        return np.array([], dtype=np.int64)
    labels, n = ndimage.label(binary)          # default structure = 4-connectivity
    sizes = np.bincount(labels.ravel())[1:]    # drop background count
    return sizes


def _parse_sources(sources: str) -> dict:
    out = {}
    for pair in sources.split(','):
        name, _, path = pair.partition('=')
        name, path = name.strip(), path.strip()
        assert name and path, f'bad --sources element: {pair!r} (expected ime=pot)'
        out[name] = Path(path)
    return out


def _plot(long_csv: Path, out_dir: Path, class_names: dict,
          out_stem: str = 'component_sizes'):
    data = {}   # (source, class) -> list of sizes
    with open(long_csv, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                key = (row['source'], int(row['class']))
                data.setdefault(key, []).append(int(row['size_px']))
            except (KeyError, ValueError):
                continue
    if not data:
        print('V CSV ni vrstic; ni kaj narisati.')
        return

    sources = sorted({k[0] for k in data})
    classes = sorted({k[1] for k in data})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'legend.fontsize': 10, 'pdf.fonttype': 42,
    })

    all_sizes = np.concatenate([np.asarray(v) for v in data.values()])
    bins = np.logspace(0, np.log10(all_sizes.max() + 1), 30)

    fig, axes = plt.subplots(1, len(classes), figsize=(5.5 * len(classes), 4),
                             squeeze=False)
    for ax, c in zip(axes[0], classes):
        for si, s in enumerate(sources):
            sizes = np.asarray(data.get((s, c), []))
            if sizes.size == 0:
                continue
            weights = np.full(sizes.shape, 1.0 / sizes.size)  # normirano na delež
            ax.hist(sizes, bins=bins, weights=weights, histtype='step',
                    linewidth=1.8, color=f'C{si}', label=s)
        ax.set_xscale('log')
        ax.set_xlabel('Velikost komponente [px]')
        ax.set_ylabel('Delež komponent')
        ax.set_title(f'{class_names.get(c, f"razred {c}")} (razred {c})')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()
    fig.suptitle('Porazdelitev velikosti povezanih komponent')
    plt.tight_layout()

    out_pdf = out_dir / f'{out_stem}.pdf'
    out_png = out_dir / f'{out_stem}.png'
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f'Shranjeno: {out_pdf}')
    print(f'Shranjeno: {out_png}')


def main(
    sources: str = 'realne=./results/bark_1024_aug_200k/eval/eval_data/masks',
    out_dir: str = './results/bark_1024_aug_200k/texture_eval',
    classes: str = '1,2',
    class_names: str = '1=slepice,2=poškodbe',
    min_size: int = 1,           # ignore components smaller than this (px)
    plot_only: bool = False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_csv = out_dir / 'component_sizes.csv'
    summary_csv = out_dir / 'component_summary.csv'

    name_map = {}
    for pair in str(class_names).split(','):
        k, _, v = pair.partition('=')
        name_map[int(k)] = v.strip()

    if plot_only:
        _plot(long_csv, out_dir, name_map)
        return

    src_map = _parse_sources(sources)
    class_list = [int(c) for c in str(classes).split(',')]
    print(f'Viri: { {k: str(v) for k, v in src_map.items()} }   razredi: {class_list}')

    rows = []          # (source, file, class, size_px)
    per_image = {}     # (source, class) -> list of per-image component counts
    for src_name, folder in src_map.items():
        files = _list_masks(folder)
        print(f'  [{src_name}] {len(files)} mask iz {folder}')
        for f in files:
            arr = np.asarray(Image.open(f))
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            for c in class_list:
                sizes = _component_sizes(arr, c)
                sizes = sizes[sizes >= min_size]
                per_image.setdefault((src_name, c), []).append(len(sizes))
                for s in sizes:
                    rows.append((src_name, f.name, c, int(s)))

    with open(long_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'file', 'class', 'size_px'])
        writer.writerows(rows)
    print(f'CSV (komponente): {long_csv}  ({len(rows)} komponent)')

    # Summary
    agg = {}
    for src_name, _, c, s in rows:
        agg.setdefault((src_name, c), []).append(s)
    with open(summary_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'class', 'class_name', 'n_components',
                         'components_per_image_mean', 'size_mean',
                         'size_median', 'size_max'])
        print(f'\n{"vir":12s} {"razred":>6s} {"št. komp.":>10s} {"komp./sliko":>12s} '
              f'{"povpr. px":>10s} {"mediana":>8s} {"maks.":>8s}')
        for (src_name, c) in sorted(per_image):
            sizes = np.asarray(agg.get((src_name, c), []), dtype=np.int64)
            counts = per_image[(src_name, c)]
            n = sizes.size
            cpi = float(np.mean(counts)) if counts else 0.0
            mean_s = float(sizes.mean()) if n else 0.0
            med_s  = float(np.median(sizes)) if n else 0.0
            max_s  = int(sizes.max()) if n else 0
            writer.writerow([src_name, c, name_map.get(c, ''), n,
                             f'{cpi:.2f}', f'{mean_s:.1f}', f'{med_s:.1f}', max_s])
            print(f'{src_name:12s} {c:6d} {n:10d} {cpi:12.2f} '
                  f'{mean_s:10.1f} {med_s:8.1f} {max_s:8d}')
    print(f'\nCSV (povzetek): {summary_csv}')

    _plot(long_csv, out_dir, name_map)


if __name__ == '__main__':
    fire.Fire(main)
