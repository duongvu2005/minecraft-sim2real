# minecraft-sim2real

Code for the paper *"[Minecraft for Semantic Segmentation: A Procedural Data Pipeline and
Sim-to-Real Analysis]"* (MIT 6.S058, 2026).

A pipeline for generating semantic segmentation training data from procedural Minecraft worlds, and an empirical study of low-fidelity sim-to-real transfer to Cityscapes.

## Overview

The pipeline has three stages:

1. **World generation** — [Arnis](https://github.com/louis-e/arnis) converts OpenStreetMap data into Minecraft worlds whose geometry mirrors a real city.
2. **Paired capture** — at each viewpoint, two screenshots are taken: one with the default texture pack (RGB image) and one with a custom flat-color resource pack (segmentation mask).
3. **Label post-processing** — raw color masks are converted to integer-valued class maps via a per-pixel channel-dominance classifier.

The output is a directory of paired `(rgb_NNNN.png, mask_NNNN.png)` files, drop-in compatible with any standard semantic segmentation training pipeline.

## Repository structure

```
minecraft-sim2real/
├── README.md
├── LICENSE
├── .gitignore
├── pipeline/
│   ├── make_resource_pack.py
│   └── process_screenshots.py
├── notebooks/
│   └── segformer_minecraft_full.ipynb    # everything: training + eval + baseline
├── data/
│   └── example_dataset.zip
```

## Class scheme

Four classes plus an ignore label:

| Index | Class      | Color (RGB)     |
|-------|------------|-----------------|
| 0     | road       | (255, 0, 0)     |
| 1     | structure  | (0, 0, 255)     |
| 2     | vegetation | (0, 255, 0)     |
| 3     | sky        | (handled in post-processing) |
| 255   | ignore     | (excluded from loss) |

The "structure" class is the catch-all for any block that is not road, vegetation, or ignored — it aggregates buildings, walls, fences, sidewalks, and other built urban surfaces.

## Reproducing the pipeline

### 1. Generate a Minecraft world from OSM

Follow [Arnis](https://github.com/louis-e/arnis) instructions to generate a Minecraft world from a chosen OpenStreetMap bounding box. We used central Frankfurt for the experiments in the paper.

### 2. Build the segmentation resource pack

```bash
python pipeline/make_resource_pack.py
```

This extracts visible block textures from the Minecraft client files and generates a resource pack mapping each block to its semantic class.

### 3. Capture paired screenshots

Open the generated world in Minecraft and load both texture packs (default + segmentation). For each viewpoint:

1. Set time to noon: `/time set 6000`
2. F2 to capture the RGB screenshot
3. Swap to the segmentation pack
4. F2 to capture the mask
5. Swap back

Recommended graphics settings:

- Smooth Lighting: Maximum
- See-Through Leaves: Off
- Clouds: Off
- Particles: Minimal
- Brightness: Bright
- Texture Filtering: None
- Render Distance: 32 chunks

### 4. Process screenshots into paired data

```bash
python pipeline/process_screenshots.py
```

This pairs timestamped screenshots, applies the channel-dominance classifier, and writes `rgb_NNNN.png` / `mask_NNNN.png` files.

## Training and evaluation

The full training and evaluation workflow is in a single Colab notebook:

`notebooks/segformer_minecraft_full.ipynb`

Cells include (in order):
1. Setup and imports
2. Mount Drive and unzip the paired dataset
3. Train/val split (270/30, fixed seed)
4. Augmentation pipeline
5. SegFormer-B2 fine-tuning (40 epochs, ~30 min on A100)
6. Cityscapes evaluation (zero-shot transfer)
7. ADE20K baseline (same model, no Minecraft fine-tuning)
8. Confusion matrix and qualitative figure generation

Open in Colab, set runtime to GPU (A100 or L4 recommended), and run cells in order. The dataset path (cell 2) and Cityscapes path (cell 12) need to be updated for your Drive layout.

## Example dataset

A small example dataset (50 paired RGB-mask images from the Frankfurt world) is provided at `data/example_dataset.zip` for users who want to inspect the pipeline output without running the capture step themselves.

## Limitations

The dataset and pipeline released here have several known limitations, discussed in the paper:

- **Single-city collection** (Frankfurt only) limits geographic diversity
- **Manual capture** limits scale and viewpoint coverage
- **Structure aggregation** prevents distinguishing buildings from sidewalks / walls / fences
- **No vehicles, pedestrians, or urban furniture** — these categories are absent from Arnis-generated worlds and account for most of the Minecraft-to-Cityscapes transfer gap

## Citation

If you find this work useful, please cite:

```bibtex
@misc{vu2026minecraftsim2real,
  author = {Duong Vu},
  title = {Minecraft for Semantic Segmentation: A Procedural Data Pipeline and Sim-to-Real Analysis},
  year = {2026},
  howpublished = {\url{https://github.com/duongvu2005/minecraft-sim2real}},
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

This pipeline builds on:

- [Arnis](https://github.com/louis-e/arnis) — OpenStreetMap to Minecraft world conversion
- [SegFormer](https://github.com/NVlabs/SegFormer) — segmentation architecture and pretrained weights
- [Cityscapes](https://www.cityscapes-dataset.com/) — evaluation benchmark
- [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) — baseline pretraining
