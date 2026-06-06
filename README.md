# DiffInfinite-Bark

Seamlessly tileable bark-texture synthesis with a torus-topology random patch
diffusion model. Built on top of [DiffInfinite (Aversa et al., NeurIPS 2023)](https://openreview.net/forum?id=QXTjde8evS),
retargeted from lung-cancer histopathology to spruce-log bark with three semantic
classes (`bark` / `knot` / `defect`), and extended with a new toroidal
sliding-window sampler so the generated images and masks tile perfectly in both
dimensions.

This repository is part of a master's thesis project by **Jakob Kreft**. It is a
derivative work — the original codebase, paper, and intellectual contribution
are by Marco Aversa and co-authors. See [LICENSE](LICENSE) and the citations
at the bottom of this README.

![Tileable bark sample, 2×2 preview](images/showcase/torus_output_2x2_qr_code.png)

| Single tile | 2×2 tiled preview |
|---|---|
| ![tile](images/showcase/torus_output_qr_code.png) | ![tiled](images/showcase/torus_output_2x2_qr_code.png) |

A single-patch sample (`sample_torus.py --uniform_size 512`):

![single patch](images/showcase/generated_sample.png)

---

## Pretrained model and dataset

Hosted on Hugging Face:

- **Model checkpoints**: https://huggingface.co/jakobkreft/diffinfinite-bark
- **Dataset (spruce-log bark segmentation, 680 image-mask pairs at 512×512)**: https://huggingface.co/datasets/jakobkreft/spruce-log-bark-segmentation

Classes in the dataset: `bark = 0`, `knot = 1`, `defect = 2`.

---

## Install

Tested on Windows 10/11 + RTX 4000 Ada 20 GB, CUDA 11.x / 12.x.

```bash
git clone https://github.com/jakobkreft/diffinfinite-bark.git
cd diffinfinite-bark
pip install -r requirements.txt
```

`torch` and `torchvision` are intentionally unpinned — install the CUDA build
that matches your driver via https://pytorch.org/get-started/locally/ before
running `pip install -r requirements.txt`, or let pip resolve the CPU build if
you only need to read the code.

---

## Generate a tileable texture

Download the model checkpoints from Hugging Face into `./results/bark_200k/`
(the image model) and `./results/bark_masks/` (the mask model), then:

```bash
# Uniform mask, 2048x2048 tileable bark
python sample_torus.py \
    --uniform_size 2048 --uniform_class 0 \
    --results_folder ./results/bark_200k --milestone 40

# Or supply your own tileable mask PNG (single-channel, values in {0,1,2},
# dimensions multiples of 512):
python sample_torus.py --mask_path your_mask.png \
    --results_folder ./results/bark_200k --milestone 40
```

Both produce `torus_output.png` plus a `torus_output_2x2.png` 2×2 preview so
you can eyeball the seam quality.

To generate a tileable *mask* from scratch with the mask diffusion model:

```bash
python sample_torus_mask.py --image_size 2048 --label 1 \
    --results_folder ./results/bark_masks --milestone 12
```

For minimal smoke-testing on a single 512×512 patch, see
[inference-test.py](inference-test.py).

---

## Train

```bash
python train.py \
    --data_folder path/to/spruce-log-bark-segmentation \
    --results_folder ./results/bark_run \
    --train_num_steps 200000 \
    --batch_size 4 --gradient_accumulate_every 4
```

Defaults match the parameters used to produce the published checkpoints:
3 data classes + 1 unconditional CFG class, OneCycleLR, EMA, cosine
β-schedule, DDIM-250 sampling. Loss and learning-rate curves are written to
`results/<run>/training_log.csv`; plot them with `python plot_loss.py`.

To train the mask diffusion model:

```bash
python train_masks.py \
    --data_path path/to/spruce-log-bark-segmentation \
    --results_folder ./results/bark_masks \
    --train_num_steps 60000
```

---

## What's different from the original DiffInfinite

Concise summary of the substantive changes from upstream:

1. **Torus-topology sliding-window sampler.** New
   [random_diffusion_torus.py](random_diffusion_torus.py) adds
   `RandomDiffusionTorus` and `RandomDiffusionMasksTorus`, which subclass the
   upstream samplers and make the latent grid behave like a torus. Patches
   near the edges wrap to the opposite side, and the Hann-window decode uses
   `torch.roll` so the *outer* seam is blended (upstream only fixed interior
   seams). The UNet itself is unchanged.
2. **New `sample_torus.py` / `sample_torus_mask.py`** entry points for
   tileable image and mask generation, including support for rectangular
   outputs and uniform mask shortcuts.
3. **Bark dataset adapter.** [dataset.py](dataset.py) was stripped of the
   lung-cancer-specific label remapping, the `extra_unknown_data_path`
   dependency, and the empty-class-bucket bug. Subclasses are now
   auto-detected from `class_to_int.yml`.
4. **New simple mask loader.** [dataset_masks.py](dataset_masks.py) replaces
   the heavy upstream mask dataset with `SimpleMaskDataset`, which globs
   `*_mask.png` and uses the dominant class label per mask.
5. **Windows + single-GPU portability.** The unconditional `.module`
   accessors that assume DDP wrapping were replaced with `_get_vae()` and
   `hasattr(..., 'module')` checks throughout [dm.py](dm.py) and
   [dm_masks.py](dm_masks.py). `num_workers` defaults to 0 (Windows spawn
   safety). Scheduler-load is tolerant of `train_num_steps` mismatches.
6. **VAE source swap.** From the gated `stabilityai/stable-diffusion-2-base`
   to the public `stabilityai/sd-vae-ft-mse` (same architecture, MSE-tuned).
7. **CSV training log.** [dm.py](dm.py) writes `training_log.csv` with
   `(step, loss, lr)` so loss curves survive checkpoint reloads.
8. **`plot_loss.py`.** New CLI for plotting the loss / LR / per-timestep-bucket
   curves from a checkpoint.
9. **Slimmer dependencies.** [requirements.txt](requirements.txt) drops
   pyarmor / pyinstaller / pinned CUDA libs; `environment.yaml` (Linux-locked
   conda snapshot) is removed.

The original UNet, DDIM sampler, P2-weighting machinery, EMA, cosine β-schedule,
classifier-free guidance, and the base `RandomDiffusion` / `RandomDiffusionMasks`
algorithms are upstream — kept as-is.

---

## Citations

If you use this work, please cite both the original DiffInfinite paper and the
master's thesis it underpins.

The original DiffInfinite paper:

```bibtex
@inproceedings{
aversa2023diffinfinite,
title={DiffInfinite: Large Mask-Image Synthesis via Parallel Random Patch Diffusion in Histopathology},
author={Marco Aversa and Gabriel Nobis and Miriam H{\"a}gele and Kai Standvoss and Mihaela Chirica and Roderick Murray-Smith and Ahmed Alaa and Lukas Ruff and Daniela Ivanova and Wojciech Samek and Frederick Klauschen and Bruno Sanguinetti and Luis Oala},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2023},
url={https://openreview.net/forum?id=QXTjde8evS}
}
```

The thesis bibtex entry will be added here once the thesis is published.

---

## License and attribution

This repository is released under the MIT License, inheriting from the
upstream [marcoaversa/diffinfinite](https://github.com/marcoaversa/diffinfinite).
The original `Copyright (c) 2023 marcoaversa` notice is retained verbatim in
[LICENSE](LICENSE). All modifications described in *"What's different from
the original DiffInfinite"* above are © 2024–2026 Jakob Kreft, also released
under MIT.
