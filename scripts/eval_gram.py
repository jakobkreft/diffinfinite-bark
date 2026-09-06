"""Gramove matrike (VGG-19) — teksturna razdalja po Gatysu med realnimi in
generiranimi zaplatami.

Implementacija sledi Gatys et al. 2015 ("Texture Synthesis Using
Convolutional Neural Networks"), prilagojeno za metriko (ne sintezo):

    1. Vsako zaplato spustimo skozi predučen torchvision VGG-19
       (ImageNet uteži, eval način, brez gradientov). Vhod je [0,1],
       normaliziran z ImageNet povprečjem/odklonom.
    2. Na izbranih plasteh (privzeto relu1_1, relu2_1, relu3_1, relu4_1,
       relu5_1 — Gatysov "style" nabor) vzamemo značilke F oblike (C, H·W)
       in izračunamo Gramovo matriko:
           G = F · Fᵀ / (C · H · W)
       Normalizacija s številom elementov (konvencija PyTorch style-transfer
       tutoriala) naredi velikostne rede primerljive med plastmi.
    3. G povprečimo čez vse zaplate znotraj vsakega nabora (realne,
       generirane), v float64 zaradi numerične stabilnosti.
    4. Razdalja po plasti = Frobeniusova norma razlike povprečnih Gramov:
           d_l = || Ḡ_real^l − Ḡ_gen^l ||_F
       Poročamo tudi relativno različico d_l / ||Ḡ_real^l||_F, ker surove
       norme med plastmi niso primerljive (različno število kanalov).
    5. Šumno dno ("noise floor"): realni nabor razpolovimo (fiksno seme) in
       izračunamo enako razdaljo med polovicama. Če je razdalja
       real-vs-generirane blizu razdalje real-vs-real, se teksturni stil
       statistično ne loči — to je ciljni rezultat.

Rezultati se zapišejo v CSV (gram_distances.csv), graf pa se vedno lahko
obnovi samo iz CSV z --plot_only. Grafi: PDF (vektorski, za LaTeX) + PNG.
Vsi napisi na grafih so v slovenščini.

Uporaba:
    python scripts/eval_gram.py                      # privzete poti
    python scripts/eval_gram.py --real_dir ... --gen_dir ...
    python scripts/eval_gram.py --plot_only          # samo ponovno nariši
    python scripts/eval_gram.py --device cuda:1

Zahteva: torchvision (VGG-19 uteži se ob prvem zagonu prenesejo, ~550 MB).
"""
import csv
import random
import sys
from pathlib import Path

import fire
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# torchvision vgg19.features indices of the Gatys style layers
VGG19_LAYERS = {
    'relu1_1': 1,
    'relu2_1': 6,
    'relu3_1': 11,
    'relu4_1': 20,
    'relu5_1': 29,
}

IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _list_images(folder: Path):
    files = sorted(list(folder.glob('*.png')) + list(folder.glob('*.jpg')))
    assert files, f'no images in {folder}'
    return files


def _load_vgg(device: str):
    from torchvision.models import vgg19, VGG19_Weights
    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)
    return vgg


@torch.no_grad()
def _mean_grams(files, vgg, layer_indices, device, batch_size=8, label=''):
    """Mean Gram matrix per layer over a list of image files.

    Returns {layer_name: np.ndarray (C, C), float64}.
    """
    idx_to_name = {v: k for k, v in layer_indices.items()}
    max_idx = max(layer_indices.values())
    sums = {name: None for name in layer_indices}
    n = 0

    for start in range(0, len(files), batch_size):
        chunk = files[start:start + batch_size]
        imgs = torch.stack([
            IMAGENET_NORM(T.ToTensor()(Image.open(f).convert('RGB')))
            for f in chunk
        ]).to(device)

        x = imgs
        for i, layer in enumerate(vgg):
            x = layer(x)
            if i in idx_to_name:
                name = idx_to_name[i]
                F = x.flatten(2)                                   # (B, C, HW)
                C, HW = F.shape[1], F.shape[2]
                G = torch.bmm(F, F.transpose(1, 2)) / (C * HW)     # (B, C, C)
                g_sum = G.double().sum(dim=0).cpu().numpy()
                sums[name] = g_sum if sums[name] is None else sums[name] + g_sum
            if i >= max_idx:
                break
        n += len(chunk)
        print(f'\r  [{label}] {n}/{len(files)}', end='')
    print()
    return {name: s / n for name, s in sums.items()}


def _fro(a):
    return float(np.linalg.norm(a, ord='fro'))


def _plot(csv_path: Path, out_dir: Path, out_stem: str = 'gram_distances'):
    rows = []
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    'layer': row['layer'],
                    'dist_rel': float(row['dist_rel']),
                    'baseline_rel': float(row['baseline_rel']),
                })
            except (KeyError, ValueError):
                continue
    if not rows:
        print('V CSV ni vrstic; ni kaj narisati.')
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
        'legend.fontsize': 10, 'pdf.fonttype': 42,
    })

    layers = [r['layer'] for r in rows]
    d_rel  = [r['dist_rel'] for r in rows]
    b_rel  = [r['baseline_rel'] for r in rows]
    x = np.arange(len(layers))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w / 2, d_rel, w, color='C0', label='realne proti generiranim')
    ax.bar(x + w / 2, b_rel, w, color='C7', label='realne proti realnim (šumno dno)')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel('Plast VGG-19')
    ax.set_ylabel(r'Relativna Gramova razdalja  $\|\Delta G\|_F \, / \, \|G_{\mathrm{real}}\|_F$')
    ax.set_title('Gramove razdalje po plasteh VGG-19')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
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
    layers: str = 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1',
    batch_size: int = 8,
    max_images: int = None,       # optional cap per set (speed)
    seed: int = 42,               # for the real half/half baseline split
    plot_only: bool = False,
    device: str = 'cuda:0',
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'gram_distances.csv'

    if plot_only:
        _plot(csv_path, out_dir)
        return

    layer_indices = {name: VGG19_LAYERS[name] for name in layers.split(',')}

    real_files = _list_images(Path(real_dir))
    gen_files  = _list_images(Path(gen_dir))
    if max_images:
        real_files = real_files[:max_images]
        gen_files  = gen_files[:max_images]
    print(f'Realne: {len(real_files)}   Generirane: {len(gen_files)}   Plasti: {list(layer_indices)}')

    vgg = _load_vgg(device)

    # Mean Grams for the two sets
    g_real = _mean_grams(real_files, vgg, layer_indices, device, batch_size, 'realne')
    g_gen  = _mean_grams(gen_files,  vgg, layer_indices, device, batch_size, 'generirane')

    # Noise floor: real half vs half (fixed seed)
    rng = random.Random(seed)
    shuffled = real_files.copy()
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    g_half_a = _mean_grams(shuffled[:half],  vgg, layer_indices, device, batch_size, 'realne A')
    g_half_b = _mean_grams(shuffled[half:],  vgg, layer_indices, device, batch_size, 'realne B')

    # Distances + CSV
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'n_real', 'n_gen',
                         'dist_fro', 'dist_rel',
                         'baseline_fro', 'baseline_rel'])
        print(f'\n{"plast":10s} {"‖ΔG‖F":>12s} {"relativno":>10s} {"dno ‖ΔG‖F":>12s} {"dno rel.":>10s}')
        for name in layer_indices:
            ref_norm = _fro(g_real[name])
            d_fro = _fro(g_real[name] - g_gen[name])
            b_fro = _fro(g_half_a[name] - g_half_b[name])
            d_rel = d_fro / ref_norm
            b_rel = b_fro / ref_norm
            writer.writerow([name, len(real_files), len(gen_files),
                             f'{d_fro:.6e}', f'{d_rel:.6f}',
                             f'{b_fro:.6e}', f'{b_rel:.6f}'])
            print(f'{name:10s} {d_fro:12.4e} {d_rel:10.4f} {b_fro:12.4e} {b_rel:10.4f}')

    print(f'\nCSV: {csv_path}')
    _plot(csv_path, out_dir)


if __name__ == '__main__':
    fire.Fire(main)
