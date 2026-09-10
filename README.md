# NSSC 2026 · Data Analytics - Unsupervised Anomaly Detection in Mars HiRISE Imagery

This is Team OrbitIQ's Round 1 submission for the National Students' Space Challenge 2026 (IIT
Kharagpur). It finds injected "Genesis Outliers" (spliced terrain, terrestrial content, synthetic
sensor glitches) among 10,422 HiRISE crops **without labels**. The pipeline has four steps:

1. A from-scratch two-stage convolutional autoencoder compresses each crop into a 1-D latent.
2. An Isolation Forest scores each latent.
3. A statistically justified threshold decides which crops are flagged.
4. Reconstruction-error heatmaps show why each flagged crop was flagged, with a physical interpretation.

## Results at a glance

| | |
|---|---|
| **Final model** | **v6**: two-stage convolutional autoencoder trained from scratch. Stage 1 (= v3) produces a 256-d scene code (masked SSIM + L1 + gradient loss, map-projection no-data masked). Stage 2 produces a 64-d code of the stage-1 reconstruction residual. Together they form a fixed-length **320-d 1-D latent** |
| **Novelty Score** | Isolation Forest s(x, n) on the latents (500 trees × 5 seeds); higher = more anomalous |
| **Threshold** | robust modified z-score > 3.5 → **0.5251** |
| **Flagged** | **48 of 10,422 crops (0.46 %)**, in `outputs/flagged_anomalies.csv` |
| **Top-5 heatmaps** | sample_06476, sample_06963, sample_07971, sample_03383, sample_07701 |

| version | change | outcome |
|---|---|---|
| v1 | baseline CAE, pure MSE | blurry, and ~96 % of flags were map-projection border crops (ρ(novelty, no-data) = 0.63) |
| v2 | SSIM + L1 + gradient loss | blur barely changed because the bottleneck is the limit; the edge-sensitive loss made the border problem worse |
| v3 | **mask no-data wedges** | border mode gone, unimodal scores, 86 flagged; catches Earth imagery |
| v4 | spatial 8×8×4 bottleneck | **rejected**: coordinates encoded local brightness, so it flagged 510 dark or bright scenes |
| v5 | per-crop photometric normalisation | **rejected**: fakes looked more typical (benchmark AUROC 0.40) |
| v6 | **two-stage residual autoencoder** | benchmark AUROC 0.735 → **0.775**; the Earth-like images rise to ranks 1, 3, 5 and 6; 48 flagged |

**Planted-fakes benchmark** (`benchmark.py`): 2,400 synthetic splices, sensor glitches and man-made
textures are scored by each version's Isolation Forest. The forest is fitted only on the real crops. The
benchmark is used only to compare versions; the threshold rule is fixed and is never tuned on it.

| model | mean AUROC | recall at own threshold |
|---|---|---|
| raw-pixel Isolation Forest | 0.45 | 1.0 % |
| v3 | 0.735 | 9.5 % |
| **v6 (final)** | **0.775** | **12.1 %** |

Four of the six highest-scoring crops are Earth-like aerial images: an orchard, a tree canopy, runway-like
stripe markings and the Pentagon. Their reconstruction error is spread over the whole frame and is very
high. Rare genuine terrain behaves differently: its error concentrates on relief. Two north-polar
observations account for 12 of the 48 flags. Known limitations: thin streaks and dead-pixel blocks are
missed, and one aerial-buildings image ranks 156th. Details are in the report and `CHANGELOG.md`.

## Submission contents

| File | What it is |
|---|---|
| `NSSC_DA_HiRISE.ipynb` | **Main notebook** with code and outputs. Markdown headings follow the problem statement (0, 1.1-1.3, 2.1-2.4, 3.1-3.2, 4) |
| `report/NSSC_DA_Report.pdf` | **Report** in IEEE conference format (LaTeX source `report/NSSC_DA_Report.tex`), with the final answers highlighted in Table I |
| `CHANGELOG.md` | **Engineering changelog**, v1 → v6, each entry written as symptom → diagnosis → fix |
| `outputs/flagged_anomalies.csv` | Every crop above the final threshold, with its Novelty Score and source metadata |
| `outputs/novelty_scores_with_metadata.csv` | Novelty Score for all 10,422 crops, joined to source-image metadata |
| `outputs/bench_*_summary.csv` | Planted-fakes benchmark results per version |
| `outputs/figures/` | All figures used in the notebook and the report |
| `outputs/vN/` | Per-version checkpoint (`model.pt`), training history, config and Novelty Scores |
| `nssc_core.py` | Library code; the notebook contains this same code verbatim under each question heading |
| `run_version.py`, `benchmark.py`, `inspect_version.py` | Version trainer, planted-fakes benchmark and spot-check diagnostics |
| `build_notebook.py`, `notebook_text.py`, `build_report_ieee.py` | Assemble the notebook and render the IEEE PDF report |

## Reproducing

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pandas scipy scikit-learn matplotlib pillow nbformat nbconvert ipykernel gdown
jupyter nbconvert --to notebook --execute --inplace NSSC_DA_HiRISE.ipynb   # downloads the data if data/ is missing
python build_report_ieee.py                                                # needs a LaTeX install with IEEEtran
```

- **Checkpoints:** with `RETRAIN = False` (the default) the notebook loads the checkpoints in `outputs/vN/`. These were produced by the same code on an RTX 5070 Laptop GPU, in about 7 minutes per version. Set `RETRAIN = True` to train from scratch.
- **Google Colab:** open the notebook there directly. Its first code cell fetches the organisers' dataset when it isn't present.

## Rule compliance

* No pretrained weights, pretrained feature extractors, transfer learning or pretrained perceptual
  losses. Every network is randomly initialised and trained only on the competition crops.
* The Isolation Forest input is the fixed-length 1-D latent vector, not raw pixels.
* Novelty Score convention: **higher = more anomalous** (sklearn's `score_samples` is negated).
* The anomaly count comes from a statistical threshold on the observed score distribution. There is
  no top-N cutoff and no assumed contamination fraction (`contamination="auto"` never feeds into the count).
* No labels are used anywhere. The planted-fakes benchmark and the hand-noticed spot-check crops are
  used only to diagnose and compare versions, never to set the threshold.
* Metadata is joined only from the provided `source_image_metadata.csv`; nothing is scraped.
