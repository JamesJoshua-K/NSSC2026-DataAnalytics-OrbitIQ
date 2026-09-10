"""Narrative markdown for the notebook (kept separate so the builder stays readable)."""

T = {}

T["title"] = r"""
# NSSC 2026 · Data Analytics
## Unsupervised Anomaly Detection in Mars HiRISE Orbital Imagery

**Team:** OrbitIQ · National Students' Space Challenge, IIT Kharagpur

**Pipeline:** custom two-stage convolutional autoencoder (trained from scratch, no pretrained weights) →
1-D latent vector per crop → Isolation Forest Novelty Score (higher = more anomalous) →
statistically justified threshold → metadata join → pixel-wise reconstruction-error heatmaps →
physical hypotheses. Six model versions (v1 → v6) are trained and compared; the changelog
(Phase 4 and `CHANGELOG.md`) documents each symptom → diagnosis → fix step.

| Section | Content |
|---|---|
| 0 | Setup, data loading, sanity pass, no-data masking |
| 1.1 / 1.2 / 1.3 | Architecture, loss design, training of v1-v6, t-SNE of latents |
| 2.1 / 2.2 / 2.3 / 2.4 | Isolation Forest scoring, statistical threshold, location analysis, metadata fusion |
| 3.1 / 3.2 | Top-5 error heatmaps and geological report |
| 4 | Iteration journal (v1 → v6) and planted-fakes benchmark |

Reproduce: place `images.zip` (extracted to `data/images_unz/images/`), `crop_metadata_index.csv` and
`source_image_metadata.csv` in `data/`, then run all cells. With `RETRAIN=False` the notebook loads
checkpoints in `outputs/vN/model.pt` that were produced by exactly the code in this notebook.
"""

T["sanity"] = r"""
**Sanity-pass findings**

* 10,422 single-channel 227×227 JPEG crops from 172 source observations; every crop maps to exactly
  one `source_image_id` and every id has metadata (checked with `assert`).
* Pixel values span the full 0-255 range; we rescale by 1/255 to [0, 1] and use a sigmoid decoder output.
* **≈20 % of crops contain a solid-black no-data wedge** (the map-projected edge of the HiRISE strip).
  These are *normal* data, but they are the highest-contrast edges in the whole dataset, which matters
  for both the reconstruction loss and the anomaly detector (see changelog).
* Very bright crops (frost / dust) exist and are genuine terrain, so a brightness heuristic
  cannot separate anomalies - consistent with the organisers' warning.
"""

T["nodata"] = r"""
v1 and v2 showed that the black map-projection wedges, which are *normal* data, dominated both the
reconstruction loss and the latent space (see changelog). From v3 on, each crop gets a **no-data
mask**. A pixel is masked if it belongs to a near-black (≤ 5) region that

* survives a 5×5 morphological opening, so thin dark lines are *not* masked. A dead sensor
  column or a black streak could itself be an injected anomaly, and hiding it would be wrong.
* touches the crop border, because strip edges enter from outside the crop.
* covers at least 0.5 % of the crop.

The mask is dilated by 2 px to cover JPEG ringing along the wedge edge. Masked pixels are filled
with the median of the crop's valid pixels, which gives a flat, texture-free region with no
razor-sharp edge. They are also **excluded from the loss and from every reconstruction-error
metric and heatmap**. The filled image, not the original, is what the v3 encoder sees.
"""

T["arch"] = r"""
**Design.** A convolutional autoencoder written from scratch in PyTorch (no pretrained weights,
no transfer learning).

* **Encoder:** five stride-2 3×3 convolutions (32→64→128→256→256 channels), BatchNorm + LeakyReLU.
  227 → 114 → 57 → 29 → 15 → 8. Five halvings are the minimum that brings the map to 8×8, so the
  deepest units have a receptive field covering a large part of the crop (whole craters, dune trains)
  before the bottleneck.
* **Stage-1 bottleneck (v3, reused unchanged as stage 1 of the final model v6):** flatten (256×8×8 = 16,384) → `Linear` → **fixed-length 1-D
  latent vector of 256 floats** (the Phase 2 input). The global linear layer lets every latent
  coordinate combine evidence from the whole crop, so the code describes *what kind of scene* this
  is (terrain type, texture class, relief).
* **Alternative tried and rejected (v4):** a *spatial* bottleneck, a 1×1 convolution to an 8×8
  grid × 4 channels flattened to the same 256-d vector. The aim was to keep small local defects
  visible, but each coordinate then mostly encoded *local brightness*, so the Isolation Forest
  isolated very bright or dark scenes instead of anomalies (see changelog). We therefore kept the
  global code.
* **Final model (v6): two-stage autoencoder.** A second autoencoder of the same design, with a
  64-float bottleneck, is trained on stage 1's amplified signed reconstruction residual,
  128 + 127·clip(4·(x − x̂), −1, 1). The final latent is the **fixed-length 1-D vector
  [256-d scene code | 64-d residual code] = 320 floats**. The scene code says *what kind of terrain*
  the crop is; the residual code says *what the scene code could not explain*, which is where splices,
  glitches and foreign content live. Both stages are trained from scratch on the competition crops only.
* **Decoder:** mirror image: `Linear` (v4: 1×1 conv) back to the 8×8×256 map → five ×2
  transposed-conv stages to 256×256, centre-crop to 227×227, sigmoid. Cropping (rather than odd
  output-padding) keeps every stage a clean ×2.
* Stride-2 convolutions replace pooling so the network learns its own downsampling.
* The bottleneck type, latent size, loss and no-data handling are configurable because they change
  across v1 → v4 (`CONFIGS`).
"""

T["latent_sweep"] = r"""
The latent size trades **reconstruction fidelity** (a larger code keeps more detail) against
**outlier separability**. An Isolation Forest splits on randomly chosen coordinates. If many latent
dimensions carry redundant or near-constant information, most random splits are wasted and the
isolation depth of true outliers gets diluted. We trained the stage-1 configuration (the scene code of the final model) with d ∈ {64, 128,
256, 512} (short schedule) and measured three things:

* held-out SSIM (fidelity)
* the number of principal components needed for 90 % of latent variance (how much of the code is used)
* a label-free separability proxy: the *tail ratio* (99.9th-percentile Novelty Score − median) / MAD.
  This measures how far the most isolated crops sit from the bulk, in robust units.
"""

T["loss"] = r"""
**Loss design.** v1 used pure MSE. MSE is minimised by the conditional mean, so when fine texture
(dune ripples, crater rims, polygonal ground) is uncertain the decoder outputs a blurred average.
The v1 reconstructions show this clearly (Fig 1.2). From v2 on, the objective is
**0.84·(1 − SSIM) + 0.16·L1 + 0.1·Gradient-L1**, computed only on valid (non-no-data) pixels from v3 on.
Its fully differentiable terms (weights in `CONFIGS`) complement each other:

* **(1 − SSIM)** with an 11×11 Gaussian window (σ = 1.5) compares local mean, contrast and structure.
  It penalises the loss of local structure even when the average brightness is right, which directly
  targets blur on ridges and crater rims.
* **L1** anchors absolute brightness. SSIM is insensitive to uniform intensity offsets, so on
  its own it can drift in albedo.
* **Gradient L1** (finite-difference edges) penalises smeared or misplaced high-frequency edges,
  which helps sharp linear features such as ridgelines and scarps.

No perceptual loss is used, because every perceptual network we could use would be pretrained, and
that is banned.

**What the iterations taught us about this loss.** Switching from MSE (v1) to this combination (v2)
sharpened crater rims and scarps, but it did **not** recover dune-ripple texture: held-out SSIM only
went from 0.7053 to 0.7065. The binding constraint is the bottleneck, not the loss. 256 floats for
51,529 pixels cannot store the exact phase of a ripple field, and a loss cannot recreate information
the code has discarded. We keep the structural loss because it spends the unavoidable error on
texture detail rather than on structure. Its edge sensitivity, however, made the network lock onto
the razor-sharp edges of the no-data wedges (v2's novelty correlated 0.66 with the no-data share).
That is why, from v3 on, every term is computed on valid pixels only (Section 0.2).

**Stage-2 loss (v6).** The residual autoencoder reconstructs a signed residual image, not terrain, so
structural similarity is not meaningful there; it uses MSE + 0.5·L1 on valid pixels.
"""

T["tsne_intro"] = r"""
PCA (50 components on standardised latents) followed by t-SNE (perplexity 40). Colouring by simple
image properties the model never saw as labels shows *which* factors of variation the latent space
organises. This is a qualitative diagnostic only, not proof of geological clusters.
"""

T["if_intro"] = r"""
The Isolation Forest is fitted on the 1-D latent vectors of **all** crops (unsupervised: there is no
"clean" training set). We use 500 trees and 256-sample sub-sampling (the original paper's
recommended ψ), and average 5 forests with different seeds to reduce the variance of the ranking.

**Score convention.** sklearn's `score_samples` returns the *negated* anomaly score (higher = more
normal). We negate it, so our **Novelty Score = s(x, n) ∈ (0, 1] from Liu et al. (2008), and higher
= more anomalous**. This is the competition convention, and it is verified below by the sign of the
correlation with sklearn's `decision_function`.
"""

T["thr_intro"] = r"""
We want the boundary to come from the **shape of the observed score distribution**, not from an
assumed count. First we test whether a Gaussian μ + kσ rule is even appropriate (skewness, kurtosis,
D'Agostino K² test, Q-Q plot). Then we compare three distribution-driven rules:

1. **Robust modified z-score** (Iglewicz & Hoaglin 1993): M = 0.6745 (s − median) / MAD, flag
   M > 3.5. Median and MAD have a 50 % breakdown point, so the outliers being hunted cannot inflate
   the threshold. Under normality 0.6745 × MAD estimates σ, so 3.5 is comparable to ≈3.5σ.
2. **Kneedle elbow** of the ascending sorted-score curve: min-max normalise both axes and take the
   point of maximum vertical distance below the chord joining the end points. This is the point of
   maximum curvature, where the curve turns from "bulk" to "tail".
3. **Two-component Gaussian mixture**: the score where the posterior of the high-mean component
   exceeds 0.5.
"""

T["thr_choice"] = r"""
### Choice: robust modified z-score > 3.5 → **threshold 0.5251, 48 crops flagged (0.46 %)**

1. **A Gaussian μ + kσ rule is not supported by the data.** The final (v6) scores have skewness 0.75,
   excess kurtosis 0.65 and D'Agostino K² = 885 (p ≈ 0), and the Q-Q plot bends upward in the right
   tail. μ and σ are also estimated *including* the tail we are trying to find, so the outliers
   inflate their own threshold. We keep μ + 3σ only as a cross-check.
2. **The elbow and GMM boundaries sit inside the body of the distribution, not at a break.** The score
   distribution is unimodal. A two-component GMM therefore splits the single skewed bulk into two
   overlapping Gaussians, and its boundary falls near the 64th percentile (3,740 crops). The Kneedle
   elbow of the *whole* sorted curve marks where the curve leaves its straight middle section, at the
   start of the ordinary right skew of normal terrain (987 crops). Neither boundary coincides with any
   change in the behaviour of the density.
3. **The modified z-score is built for exactly this situation.** Median and MAD have a 50 % breakdown
   point, so however many outliers exist they cannot move the threshold. k = 3.5 is the value
   recommended by Iglewicz & Hoaglin (1993). Under a Gaussian bulk with this robust scale about 2.4 of
   10,422 normal crops would exceed it by chance; the real bulk is right-skewed, so this is a lower
   bound, not a guarantee.
4. **Cross-checks bracket the choice.** μ + 3σ flags 82 crops, and the same robust rule applied to
   log-scores (which reduces the skew from 0.75 to 0.54) flags 22. The chosen 48 lies between the
   classical and the skew-corrected rule, and on the sorted-score plot all three sit in the final
   steep rise, where the curve turns from gradual to near-vertical.
5. **Sensitivity is reported, not tuned.** k = 3, 3.5, 4 and 5 give 97, 48, 24 and 1 flagged crops.
   We keep the conventional k = 3.5 rather than choosing k to hit a count.
"""

T["thr_code"] = r'''
THRESHOLD = t_mad            # final rule: robust modified z-score > 3.5 (justification above)
flagged = nov > THRESHOLD
print(f"FINAL NOVELTY THRESHOLD = {THRESHOLD:.4f}  ->  {flagged.sum()} of {len(nov)} crops flagged ({flagged.mean():.2%})")
for k in (3.0, 3.5, 4.0, 5.0):          # sensitivity of the count to the cutoff constant (reported, not tuned)
    print(f"  k = {k}: threshold {mad_threshold(nov, k):.4f} -> {(nov > mad_threshold(nov, k)).sum()} flagged")
print(f"expected chance exceedances if the bulk were Gaussian with the robust scale: {len(nov) * stats.norm.sf(3.5):.1f} crops")
lt = np.log(nov)                          # skew-robust cross-check: the same rule on log-scores
print(f"same rule on log-scores (skew {stats.skew(lt):.2f}): {(lt > mad_threshold(lt, 3.5)).sum()} flagged")
'''

T["contamination"] = r"""
sklearn's `IsolationForest(contamination=...)` only changes the offset used by `predict()` to produce
±1 labels. It has **no effect on `score_samples()`**, which is all we use. We set
`contamination="auto"` (sklearn's default, offset −0.5), and we never call `predict()`. The final
anomaly count is determined solely by the statistical threshold on the continuous Novelty Score
above, **independently of any assumed contamination fraction**.
"""

T["loc_intro"] = r"""
Each scored crop is joined to `source_image_metadata.csv` through its `source_image_id` (from
`crop_metadata_index.csv`), many-to-one and validated. We then ask whether flags cluster in
particular observations, latitudes, illumination conditions, seasons or resolutions.

**Metadata resolution.** The provided coordinates are quantised. Latitude is on a 5° grid from −45°
to 60°, plus the two poles (±90°), and longitude takes only two values, 0° or 180°E (29 and 143
sources). A longitude-latitude map would collapse to two vertical lines, so we analyse
**latitude bands × the two longitudes** instead of point locations. We test
per-source enrichment with a one-sided binomial test against the global flag rate
(Bonferroni-corrected), and test trends with rank statistics (Spearman, Kruskal-Wallis), which
make no Gaussian assumption.
"""

T["fusion_intro"] = r"""
We concatenate the standardised image latent with an encoded metadata vector: z-scored sun angle,
sin/cos of latitude and longitude (to respect circularity), log₂ resolution, and a one-hot season.
We re-score with the same Isolation Forest settings and the same modified-z rule, then compare the
flagged sets. The image-only pipeline remains the primary result.
"""

T["hm_intro"] = r"""
The five highest-scoring crops **within the statistically thresholded set** are passed back through
the Phase 1 decoder; for the two-stage final model this is stage 1's decoder, i.e. the image-space
reconstruction. The per-pixel squared error (validity-weighted smoothing, σ = 2 px) is shown alone and
overlaid on the original. The validity map is eroded by 3 px so JPEG ringing at a filled no-data
boundary is not read as error, and each map uses a robust colour scale (99.5th percentile of valid
pixels). To make the physical reading less subjective, we also compute descriptors of each error map
and image (`heatmap_descriptors`), each reported as a percentile against 500 random un-flagged crops:

* error **concentration** (share of error in the top 5 % of pixels)
* **line score** (a single-row or single-column error spike, as a seam would produce)
* **seam step** (the largest mean-intensity jump across a straight line)
* spectral **periodicity** (stripes or banding)
* **saturation** (clipped pixels)
"""

T["changelog_intro"] = r"""
Every version was trained with identical data, schedule and seed. The table below is the
quantitative evidence referenced in each changelog entry, followed by the planted-fakes benchmark.
The full journal is also in `CHANGELOG.md`.

Note: `val_mse` / `val_ssim` of v1-v2 include no-data pixels, whereas v3 onwards are computed on valid
pixels only, so those columns are not strictly like-for-like. v6 reports its stage-1 (= v3) values.
"""

# ---------------------------------------------------------------- result-dependent narrative (v3 run)
T["latent_choice"] = r"""
**Choice: d = 256.** The sweep uses a short 12-epoch schedule and a single seed, so its flagged
counts are not comparable with the fully trained model and are shown only for completeness. The
differences are indicative, not conclusive.

* **Fidelity saturates.** Held-out SSIM is 0.682 (64), 0.678 (128), 0.692 (256) and 0.696 (512).
  Doubling from 256 to 512 buys only +0.004.
* **Effective dimensionality grows sub-linearly.** 30, 8, 85 and 109 principal components hold 90 %
  of the latent variance. Going to 512 adds 256 coordinates but only about 24 effective directions.
  The rest are redundant, and redundant coordinates **dilute an Isolation Forest**, whose splits pick
  a coordinate uniformly at random.
* **The separability proxy is flat.** The tail ratio is 10.9, 9.5, 11.0 and 10.5, too close to rank the
  candidates; it also rewards nuisance modes (the rejected v2 scores 14.7), so it is only a sanity check. The correlation with the
  no-data fraction (the nuisance removed in v3) creeps up with size (0.03 → 0.08), because a bigger
  code has room to re-encode fill geometry.
* **d = 128 is an outlier.** 90 % of its variance sits in only 8 components, a partial collapse
  under the short schedule, and it has the weakest tail.

We therefore choose 256 on fidelity and effective dimensionality: it is the smallest code at which
reconstruction fidelity is essentially saturated without adding largely redundant coordinates.
"""
T["tsne_read"] = r"""
**Reading the map (visual interpretation, not labels).**

* **No collapse.** The projection is not one structureless blob. It has a dense core, extended arms and
  several small isolated islands.
* **Brightness is the dominant organising axis.** Mean intensity changes smoothly from one side of the
  map to the other; contrast varies more locally.
* **Pixel scale is encoded too.** Crops from the single 1 m/px observation concentrate in one compact
  region, and 0.5 m/px crops are over-represented in parts of the map. The same terrain imaged at a
  coarser scale has a different texture frequency.
* **The no-data fraction is spread uniformly.** This is visual confirmation that masking removed the
  border nuisance that dominated v1 and v2.
* The thumbnails show coherent neighbourhoods rather than discrete clusters (knobby and rough terrain,
  cratered plains, ripples and dunes, bright etched or polygonal ground, smooth dusty slopes): a
  continuum of terrain types.
* After thresholding (Fig 1.3c, produced in Section 2.2), flagged crops lie mainly on the **periphery
  of the map and in its outer lobes**, with only a few in the interior, as expected if the Isolation
  Forest isolates sparse regions.
"""

T["loc_read"] = r"""
**Findings (final model v6).**

* The 48 flags come from 30 of the 172 source observations. **Two sources are significantly
  enriched** (one-sided binomial test against the 0.46 % base rate, Bonferroni-corrected p < 0.05),
  and both are in the **90° N metadata bin**: SRC_039 (9 of its 16 crops flagged) and SRC_166 (3 of 16).
  Together they hold 12 of the 48 flags.
* **Physical reading: most likely rare terrain, not tampering.** These crops show bright ice-rich
  surfaces, finely banded layered deposits and scarps (e.g. the "stripes" of 08731 are layering, not
  sensor banding). This terrain is a small minority of the training crops, so its latents are easy to
  isolate. A mission pipeline would *route* such flags to the polar team rather than discard them; the
  metadata is what identifies them as a terrain class. Latitude is binned (the grid jumps from 60° to a
  90° bin), so "90° N" is a bin, not the pole itself; the 90° S bin has no flags.
* **The injected content is not location-specific.** The four Earth-like crops in the top 6 (06476,
  07971, 07701, 09828) come from four different observations between 35° S and 15° N, as expected if
  foreign crops were inserted at random positions.
* **Latitude.** Outside the 90° N bin the flag rate fluctuates around the base rate (0-1.1 %).
  Correlation with |latitude| is negligible (ρ = 0.06).
* **Illumination.** ρ(sun angle, novelty) = −0.03: detectable only because n = 10,422 and negligible in
  size; flagged crops span 36-79° of the dataset's 32-86° range.
* **Season.** Score *distributions* differ slightly between seasons (Kruskal-Wallis p ≈ 6·10⁻⁵), but
  *flagging* is independent of season (χ² p = 0.53).
* **Resolution.** Scores differ by resolution (p ≈ 10⁻⁶), consistent with the t-SNE observation that
  pixel scale changes texture frequency. The lone 1 m/px observation (SRC_111) has 2 of 137 crops
  flagged, so scale shifts the bulk of the distribution but does not by itself produce flags.
"""

T["fusion_read"] = r"""
Fusion changes little: 54 crops are flagged against 48 image-only, with 46 in common (Jaccard 0.82)
and a rank correlation of 0.993 between the two scorings; fusion adds 8 crops and drops 2. Two
reasons explain the small effect:

1. The Isolation Forest picks split features uniformly at random, and only 10 of the 330 fused
   dimensions are metadata, so metadata drives about 3 % of splits.
2. Every crop from one observation shares identical metadata. Metadata can therefore isolate an
   unusual *observation*, but never a single crop within it, whereas the injected anomalies are
   single crops.

Metadata is therefore most useful as we used it in 2.3, to *explain* flags after the fact (e.g. the
polar enrichment), rather than as an input to detection.
"""

T["geo_report"] = r"""
These are **hypotheses**, argued from the heatmap pattern, the descriptors (with percentiles against
500 normal crops) and the metadata. They are not definitive classifications. A quantitative pattern
runs through the top 5: the three crops whose error is **diffuse** (concentration at the 2nd-5th
percentile) are exactly the three with **very high** reconstruction error (0.016-0.030, against
0.003-0.005 for the other two), and all three look man-made.

**#1 sample_06476 (SRC_158, 5° S): straight rows of rounded, shadow-casting objects. Hypothesis:
terrestrial content (an orchard or plantation seen from above).** A near-perfect lattice of rounded
blobs in straight rows is a geometry no Martian process produces at this scale. The decoder, which has
only a Martian vocabulary, returns a smeared stripe pattern; the error covers **every blob in the
frame** (concentration 0.16, 2nd percentile) and the reconstruction error is at the 100th percentile of
the dataset (0.0295). This is the textbook **non-Martian domain-shift** signature.

**#2 sample_06963 (SRC_166, 90° N bin): bright plain with dark, angular knobs. Hypothesis: genuine
polar terrain.** A bright, smooth surface carries scattered dark knobs and a dark scarp margin at the
bottom. The error is **concentrated on the dark relief** (concentration at the 88th percentile) and
the reconstruction error is low (0.0051), so the decoder understands the scene and only misses the
sharp dark features. SRC_166 is one of the two significantly enriched polar observations (2.3).

**#3 sample_07971 (SRC_078, 15° N): densely packed rounded crowns. Hypothesis: terrestrial content
(tree canopy or orchard).** Tightly packed, rounded, textured crowns with dark gaps between them form a
loose grid. As for #1, the error is **spread across the whole frame** (concentration 0.18, 5th
percentile) and the reconstruction error is very high (0.0157), with the decoder falling back to a
blurred cratered texture. Diffuse whole-frame failure again points to a **domain shift** rather than
an unusual Martian feature.

**#4 sample_03383 (SRC_067, 15° S): dark, low-sun rugged relief. Hypothesis: genuine massifs or ridges
under oblique illumination.** The frame is darker than 99 % of reference crops, its contrast is at the
99th percentile, and the error is concentrated on the **thin sunlit crests** (concentration at the
96th percentile), not in the shadows. Long shadows from steep spurs make a high-contrast,
high-frequency pattern whose exact crest geometry the code cannot store: rare, but Martian.

**#5 sample_07701 (SRC_136, 35° S): evenly spaced bright bars inside rectangles, beside straight
lines. Hypothesis: terrestrial content (runway or apron markings) or a synthetic insertion.** Two
rectangular blocks of parallel bright bars bounded by straight edges sit next to a straight bright band
and a thin line network: rectilinear, periodic, painted-looking geometry. The error follows **every bar
and edge** (concentration 0.16, 3rd percentile; seam step at the 100th percentile because of the sharp
straight boundaries) and the reconstruction error is high (0.0170). The regularity and the diffuse
failure match the domain-shift signature of #1 and #3. v3 flagged this crop but ranked it much lower;
v6's residual code lifted it into the top 5.

**Beyond the top 5 (Fig 3.1b, diagnostic).** The **Pentagon** aerial photograph ranks 6th, with
reconstruction error at the 100th percentile spread diffusely over the frame (concentration 0.18). The
**black bar** and **two streaks** are not flagged (ranks 344 and 2,517) but sit at the 99.4th and
98.6th percentiles of reconstruction error, with the error lying *along the straight dark defect* (line
scores 4.6 and 5.4, the highest in the table): the **synthetic sensor-artefact** signature, still
partly hidden from the latent. The dead-pixel blocks are invisible even to the residual (52nd
percentile).

| heatmap pattern | examples | physical hypothesis |
|---|---|---|
| intense error over the whole frame + very high reconstruction error | 06476, 07971, 07701, Pentagon 09828 | domain shift: non-Martian (man-made) content |
| error concentrated on sharp relief, low reconstruction error | 06963, 03383 | rare genuine terrain (polar knobs, low-sun ridges) |
| error along a straight dark line | bar 00015, streaks 00503 | synthetic sensor artefact (missed by the latent) |
"""

T["final"] = r"""
<div style="background:#fff6d6;border:2px solid #d9a400;border-radius:6px;padding:10px 16px">

**FINAL ANSWERS**

| | |
|---|---|
| **Final model** | **v6**: two-stage convolutional autoencoder trained from scratch. Stage 1 (= v3): global linear bottleneck → 256-d scene code, loss 0.84·(1−SSIM) + 0.16·L1 + 0.1·Gradient-L1 on valid pixels, map-projection no-data masked. Stage 2: 64-d code of the stage-1 reconstruction residual. Final fixed-length **320-d 1-D latent** |
| **Novelty Score** | Isolation Forest s(x, n) on the latents (500 trees × 5 seeds); **higher = more anomalous** |
| **Statistical threshold** | robust modified z-score > 3.5 (median + 3.5·MAD/0.6745) = **0.5251** |
| **Crops flagged** | **48 of 10,422 (0.46 %)**, listed in `outputs/flagged_anomalies.csv` |
| **Top-5 (Phase 3)** | sample_06476, sample_06963, sample_07971, sample_03383, sample_07701 |
| **Location (2.3)** | 2 of 172 observations significantly enriched, both in the 90° N bin (12 of the 48 flags); the Earth-like crops come from four different observations |
| **Fusion (2.4)** | 54 flagged with metadata vs 48 image-only (Jaccard 0.82): metadata explains flags rather than finding them |
| **Heatmaps (3.2)** | diffuse whole-frame error + very high reconstruction error = non-Martian content (3 of the top 5, plus the Pentagon at #6); error on sharp relief = rare genuine terrain; straight-line error = seam / glitch |

</div>

**Key findings**

1. **The decisive data fix** was masking the map-projection no-data wedges (v3). It removed a nuisance
   mode that had made about 96 % of v1/v2 flags border crops, and turned a bimodal score distribution
   into a unimodal one with a genuine tail.
2. **The decisive representation fix** was the two-stage residual latent (v6). On 2,400 planted fakes
   it raised the mean AUROC from 0.735 to 0.775 (raw pixels: 0.45), made novelty agree more with
   reconstruction failure (ρ 0.44 → 0.57) and reduced the brightness bias. The Earth-like images now
   rank 1, 3, 5 and 6.
3. Two versions were tried and **rejected**: a spatial bottleneck (v4; its coordinates encoded local
   brightness) and per-crop photometric normalisation (v5; it made fakes look more typical).
4. **Limitations:** thin streaks and dead-pixel blocks are still missed by latent-space scoring, one
   aerial-buildings image (05508) ranks 156th, and rare polar terrain competes with injected content in
   the tail.
"""

T["bench_intro"] = r"""
### Planted-fakes benchmark (model-selection evidence)

Without ground truth a model cannot be scored directly. `benchmark.py` therefore plants 2,400 synthetic
anomalies into clean crops: 300 each of eight types, namely straight-seam and patch splices; bars,
blocks, gain stripes and line dropouts; and periodic and rectilinear man-made textures. The fakes pass
through exactly the same pre-processing as real crops and are scored by each version's Isolation Forest
fitted on the **real** crops only. AUROC measures how well a version ranks fakes above real crops
(0.5 = chance); recall is the share caught by that version's own statistical threshold. The threshold
rule is **never** tuned on this benchmark. It is used only to compare versions, and it is a proxy,
because our generators may differ from the organisers' injections.
"""
