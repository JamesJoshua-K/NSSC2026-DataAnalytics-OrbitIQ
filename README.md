# NSSC 2026 · Data Analytics - Unsupervised Anomaly Detection in Mars HiRISE Imagery

This is our Round 1 submission for the National Students' Space Challenge 2026 (IIT Kharagpur). It
finds injected "Genesis Outliers" (spliced terrain, terrestrial content, synthetic sensor
glitches) among 10,422 HiRISE crops **without labels**. The pipeline is a from-scratch
convolutional autoencoder, then an Isolation Forest on its latent space, then a statistically
justified threshold, then reconstruction-error heatmaps with a physical interpretation of each.

## Results at a glance

| | |
|---|---|
| **Final model** | v3: convolutional autoencoder trained from scratch; global linear bottleneck → fixed-length 256-d 1-D latent; loss 0.84·(1−SSIM) + 0.16·L1 + 0.1·Gradient-L1; map-projection no-data wedges masked |
| **Novelty Score** | Isolation Forest s(x, n) on the latents (500 trees × 5 seeds); higher = more anomalous |
| **Threshold** | robust modified z-score > 3.5 → **0.5293** |
| **Flagged** | **86 of 10,422 crops (0.83 %)**, in `outputs/flagged_anomalies.csv` |
| **Top-5 heatmaps** | sample_00849, sample_06917, sample_05849, sample_03423, sample_03383 |

| version | change | outcome |
|---|---|---|
| v1 | baseline CAE, pure MSE | blurry, and **every flag was a map-projection border crop** (ρ(novelty, no-data) = 0.63) |
| v2 | SSIM + L1 + gradient loss | blur barely changed (the bottleneck is the limit); edge-sensitive loss made the border problem worse (ρ = 0.66) |
| v3 | **mask no-data wedges** (fill + exclude from loss) | border mode gone (ρ = 0.11), unimodal scores, 86 flagged; catches Earth imagery (Pentagon, orchard grid) |
| v4 | spatial 8×8×4 bottleneck | **rejected**: coordinates encoded local brightness, so it flagged 510 dark or bright scenes |

The flagged set mixes clear **non-Martian content** with **rare genuine terrain**. The content
anomalies show diffuse whole-frame reconstruction error; the rare terrain is mainly north-polar
layered deposits, and two polar observations account for 19 of the 86 flags. Known limitations:
small local sensor defects are missed by latent-space scoring, and recall for Earth imagery is
incomplete. Details are in the report and `CHANGELOG.md`.

## Submission contents

| File | What it is |
|---|---|
| `NSSC_DA_HiRISE.ipynb` | **Main notebook** with code and outputs; markdown headings follow the problem statement (0, 1.1-1.3, 2.1-2.4, 3.1-3.2, 4) |
| `report/NSSC_DA_Report.pdf` | **Report**: explanations, labelled figures, tables, thresholds; final answers highlighted |
| `CHANGELOG.md` | **Engineering changelog**, v1 → v2 → v3 → v4, each entry written as symptom → diagnosis → fix |
| `outputs/flagged_anomalies.csv` | Every crop above the final threshold, with Novelty Score and source metadata |
| `outputs/novelty_scores_with_metadata.csv` | Novelty Score for all 10,422 crops joined to source-image metadata |
| `outputs/figures/` | All figures used in the notebook and report |
| `outputs/vN/` | Per-version checkpoint (`model.pt`), training history, config, Novelty Scores |
| `nssc_core.py` | Library code; the notebook contains the same code verbatim under each question heading |
| `run_version.py` | Command-line trainer used for the version experiments and the latent-size sweep |
| `inspect_version.py` | Spot-check diagnostics: rank of visually noticed crops per version, plus flagged-crop grids (`outputs/vN_flagged_grid.png`) |
| `build_notebook.py`, `notebook_text.py`, `build_report.py` | Assemble the notebook and render the PDF report |

## Reproducing

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scipy scikit-learn matplotlib pillow nbformat nbconvert ipykernel markdown
# data/ must contain crop_metadata_index.csv, source_image_metadata.csv and images.zip
# extracted to data/images_unz/images/
jupyter nbconvert --to notebook --execute --inplace NSSC_DA_HiRISE.ipynb
```

`RETRAIN = False` (the default) loads the checkpoints in `outputs/vN/`. These were produced by the
same code, on an RTX 5070 Laptop GPU, in about 7 min per version. Set `RETRAIN = True` to train
from scratch.

## Rule compliance

* No pretrained weights, pretrained feature extractors, transfer learning, or pretrained perceptual
  losses. Every network is initialised randomly and trained only on the competition crops.
* The Isolation Forest input is the fixed-length 1-D latent vector, not raw pixels.
* Novelty Score convention: **higher = more anomalous** (sklearn's `score_samples` is negated).
* The anomaly count comes from a statistical threshold on the observed score distribution. We use no
  top-N cutoff and no assumed contamination fraction (`contamination="auto"` never feeds into the
  count).
* Metadata is joined only from the provided `source_image_metadata.csv`; nothing is scraped.
