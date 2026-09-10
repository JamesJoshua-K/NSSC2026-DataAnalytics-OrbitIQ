"""Narrative markdown for the notebook (kept separate so the builder stays readable)."""

T = {}

T["title"] = r"""
# NSSC 2026 · Data Analytics
## Unsupervised Anomaly Detection in Mars HiRISE Orbital Imagery

**Team:** OrbitIQ · National Students' Space Challenge, IIT Kharagpur

**Pipeline:** custom convolutional autoencoder (trained from scratch, no pretrained weights) →
1-D latent vector per crop → Isolation Forest Novelty Score (higher = more anomalous) →
statistically justified threshold → metadata join → pixel-wise reconstruction-error heatmaps →
physical hypotheses. Four model versions (v1 → v4) are trained and compared; the changelog
(Phase 4 and `CHANGELOG.md`) documents each symptom → diagnosis → fix step.

| Section | Content |
|---|---|
| 0 | Setup, data loading, sanity pass, no-data masking |
| 1.1 / 1.2 / 1.3 | Architecture, loss design, training of v1-v4, t-SNE of latents |
| 2.1 / 2.2 / 2.3 / 2.4 | Isolation Forest scoring, statistical threshold, location analysis, metadata fusion |
| 3.1 / 3.2 | Top-5 error heatmaps and geological report |
| 4 | Iteration journal (v1 → v2 → v3 → v4) |

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
* **Bottleneck (final model, v3):** flatten (256×8×8 = 16,384) → `Linear` → **fixed-length 1-D
  latent vector of 256 floats** (the Phase 2 input). The global linear layer lets every latent
  coordinate combine evidence from the whole crop, so the code describes *what kind of scene* this
  is (terrain type, texture class, relief).
* **Alternative tried and rejected (v4):** a *spatial* bottleneck, a 1×1 convolution to an 8×8
  grid × 4 channels flattened to the same 256-d vector. The aim was to keep small local defects
  visible, but each coordinate then mostly encoded *local brightness*, so the Isolation Forest
  isolated very bright or dark scenes instead of anomalies (see changelog). We therefore kept the
  global code.
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
isolation depth of true outliers gets diluted. We trained the final configuration with d ∈ {64, 128,
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
* **MSE / L1** anchors absolute brightness. SSIM is insensitive to uniform intensity offsets, so on
  its own it can drift in albedo.
* **Gradient L1** (finite-difference edges) penalises smeared or misplaced high-frequency edges,
  which helps sharp linear features such as ridgelines and scarps.

No perceptual loss is used, because every perceptual network we could use would be pretrained, and
that is banned.

**What the iterations taught us about this loss.** Switching from MSE (v1) to this combination (v2)
sharpened crater rims and scarps, but it did **not** recover dune-ripple texture: held-out SSIM only
went from 0.705 to 0.707. The binding constraint is the bottleneck, not the loss. 256 floats for
51,529 pixels cannot store the exact phase of a ripple field, and a loss cannot recreate information
the code has discarded. We keep the structural loss because it spends the unavoidable error on
texture detail rather than on structure. Its edge sensitivity, however, made the network lock onto
the razor-sharp edges of the no-data wedges (v2's novelty correlated 0.66 with the no-data share).
That is why, from v3 on, every term is computed on valid pixels only (Section 0.2).
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
### Choice: robust modified z-score > 3.5 → **threshold 0.5293, 86 crops flagged (0.83 %)**

1. **A Gaussian μ + kσ rule is not supported by the data.** The v3 scores have skewness 0.96,
   excess kurtosis 0.98 and D'Agostino K² = 1,363 (p ≈ 0), and the Q-Q plot bends upward in the right
   tail. Worse, μ and σ are estimated *including* the tail we are trying to find, so the outliers
   inflate their own threshold. We keep μ + 3σ only as a cross-check.
2. **The elbow and GMM boundaries sit inside the body of the distribution, not at a break.** After
   v3 removed the no-data mode, the score distribution is **unimodal**. A two-component GMM
   therefore splits the single skewed bulk into two overlapping Gaussians, and its boundary falls
   near the 71st percentile (2,986 crops). The Kneedle elbow of the *whole* sorted curve marks where
   the curve leaves its straight middle section, at the start of the ordinary right skew of normal
   terrain (1,513 crops). Neither boundary coincides with any change in the behaviour of the density.
3. **The modified z-score is built for exactly this situation.** Median and MAD have a 50 %
   breakdown point, so however many outliers exist they cannot move the threshold. k = 3.5 is the
   value recommended by Iglewicz & Hoaglin (1993). If the bulk were Gaussian with this robust scale,
   only about 2 of 10,422 normal crops would exceed it by chance (printed below).
4. **Independent rules agree.** The robust rule (86) and the classical μ + 3σ cross-check (95) land
   within 10 % of each other. On the sorted-score plot both sit at the foot of the final steep rise
   (the last ~100 points), where the curve turns from gradual to near-vertical. This is exactly
   the behaviour an elbow method should key on, once the elbow is sought in the tail rather than
   across the whole curve.
5. **Sensitivity is reported, not tuned.** The printout shows how the count moves for k = 3 … 5.
   We keep the conventional k = 3.5 rather than choosing k to hit a count.
"""

T["thr_code"] = r'''
THRESHOLD = t_mad            # final rule: robust modified z-score > 3.5 (justification above)
flagged = nov > THRESHOLD
print(f"FINAL NOVELTY THRESHOLD = {THRESHOLD:.4f}  ->  {flagged.sum()} of {len(nov)} crops flagged ({flagged.mean():.2%})")
for k in (3.0, 3.5, 4.0, 5.0):          # sensitivity of the count to the cutoff constant (reported, not tuned)
    print(f"  k = {k}: threshold {mad_threshold(nov, k):.4f} -> {(nov > mad_threshold(nov, k)).sum()} flagged")
print(f"expected chance exceedances if the bulk were Gaussian with the robust scale: {len(nov) * stats.norm.sf(3.5):.1f} crops")
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
the decoder. The per-pixel squared error (lightly smoothed, σ = 2 px) is shown alone and overlaid on
the original. To make the physical reading less subjective, we also compute descriptors of each
error map and image (`heatmap_descriptors`). Each descriptor is reported as a percentile against 500
random un-flagged crops:

* error **concentration** (share of error in the top 5 % of pixels)
* **line score** (a single-row or single-column error spike, as a seam would produce)
* **seam step** (the largest mean-intensity jump across a straight line)
* spectral **periodicity** (stripes or banding)
* **saturation** (clipped pixels)
"""

T["changelog_intro"] = r"""
Every version was trained with identical data, schedule and seed. The table below is the
quantitative evidence referenced in each changelog entry. The full journal is also in
`CHANGELOG.md`.
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
* **Separability peaks at 256.** The tail ratio is 10.9, 9.5, 11.0 and 10.5. The correlation with the
  no-data fraction (the nuisance removed in v3) creeps up with size (0.03 → 0.08), because a bigger
  code has room to re-encode fill geometry.
* **d = 128 is an outlier.** 90 % of its variance sits in only 8 components, a partial collapse
  under the short schedule, and it has the weakest tail.

So 256 is the knee of the trade-off: the smallest code at which reconstruction fidelity is
essentially saturated and the separability proxy is at its maximum.
"""
T["tsne_read"] = r"""
**Reading the map (visual interpretation, not labels).**

* **No collapse.** The projection is not one structureless blob. It has a dense core, extended
  arms and several small isolated islands.
* **Brightness is the dominant organising axis.** Mean intensity changes smoothly from bright
  (lower left) to dark (upper right). Contrast varies more locally.
* **Pixel scale is encoded too.** Crops from the single 1 m/px observation concentrate in one
  region at the top right, and 0.5 m/px crops are over-represented on the right-hand side. The same
  terrain imaged at a coarser scale has a different texture frequency.
* **The no-data fraction is spread uniformly.** This is visual confirmation that v3's masking
  removed the border nuisance that dominated v1 and v2.
* The thumbnails show coherent neighbourhoods rather than discrete clusters: knobby and rough
  terrain on the left, cratered plains in the centre, ripples and dunes, bright etched or polygonal
  ground, and smooth dusty slopes. It is a continuum of terrain types.
* After thresholding (Fig 1.3c, produced in Section 2.2), high Novelty Scores lie on the
  **periphery and in the small islands**, and the dense core scores low, as expected if the
  Isolation Forest isolates sparse regions. Flagged crops concentrate at the bright lower edge and
  the dark upper-right edge, which is the visible trace of v3's residual brightness bias
  (changelog, v4 entry).
"""

T["loc_read"] = r"""
**Findings.**

* Flags are spread over 39 of the 172 source observations, but **4 sources are significantly
  enriched** (one-sided binomial test against the 0.83 % base rate, Bonferroni-corrected p < 0.05).
  The two strongest are both **north-polar (90° N)**: SRC_039 has **13 of its 16 crops** flagged and
  SRC_166 has 6 of 16. Together they contribute 19 of the 86 flags. The other two are SRC_067
  (7 of 92, 15° S) and SRC_032 (3 of 14, 5° S).
* **Physical reading: rare terrain, not tampering.** North-polar crops show bright ice, finely banded
  layered deposits and scarps. For example, the "stripes" of 08731 and 03423 are polar layering, not
  sensor banding. This terrain is a small minority of the training crops, so its latents are easy
  to isolate. A mission pipeline would *route* these flags to the polar science team rather than
  discard them. The metadata is what tells us these flags are a terrain class, not artefacts.
* **Latitude.** Between 45° S and 55° N the flag rate fluctuates around the base rate (0-1.5 %), then
  jumps at 60° N (≈4.4 %) and 90° N (≈5 %). Correlation with |latitude| overall is negligible
  (ρ = 0.03), so the effect is confined to the high northern bands.
* **Illumination.** ρ(sun angle, novelty) = −0.04. It is statistically significant only because
  n = 10,422; the effect size is negligible, and flagged crops span the whole 36-79° range.
* **Season.** The score *distributions* differ slightly (Kruskal-Wallis p = 0.002), but *flagging*
  is independent of season (χ² p = 0.48).
* **Resolution.** Scores differ strongly by resolution (p ≈ 10⁻¹¹), consistent with the t-SNE
  observation that pixel scale changes texture frequency. Yet the lone 1 m/px observation (SRC_111)
  has only 2 of 137 crops flagged, so scale shifts the bulk of the distribution but does not by
  itself produce flags.
"""

T["fusion_read"] = r"""
Fusion changes little: 93 crops are flagged against 86 image-only, with 83 in common (Jaccard 0.86)
and a rank correlation of 0.992 between the two scorings. Fusion adds 10 crops and drops 3. Two
reasons explain the small effect:

1. The Isolation Forest picks split features uniformly at random, and only 10 of the 266 fused
   dimensions are metadata, so metadata drives about 4 % of splits.
2. Every crop from one observation shares identical metadata. Metadata can therefore isolate an
   unusual *observation*, but never a single crop within it, whereas the injected anomalies are
   single crops.

Metadata is therefore most useful as we used it in 2.3, to *explain* flags after the fact (e.g. the
polar enrichment), rather than as an input to detection.
"""

T["geo_report"] = r"""
These are **hypotheses**, argued from the heatmap pattern, the descriptors (with percentiles against
500 normal crops) and the metadata. They are not definitive classifications.

**#1 sample_00849 (SRC_022, 20° N): straight albedo boundary. Hypothesis: spliced terrain or a
mosaic seam.** A bright slope with gullies is cut by a *straight*, full-height boundary to a uniformly
darker band (seam step 0.42, the 100th percentile; the crop is also brighter than every reference crop).
Natural albedo contacts on Mars follow topography and are rarely ruler-straight across a whole
frame. A straight contact between two unrelated surfaces is the signature of **semantic splicing**,
or, less exotically, of a seam between two HiRISE CCD channels with mismatched gain. The
reconstruction error is low (the decoder draws both halves and the boundary), so the flag comes from
the *rarity of the configuration* in latent space, not from a failure to encode it. The only
concentrated error sits on the edge of the small filled no-data wedge at the left, so part of this
crop's score may be the residual no-data effect noted in the changelog.

**#2 sample_06917 (SRC_122, 5° S): fresh crater filling the frame. Hypothesis: genuine, rare geology.**
The error forms a **ring that follows the crater rim**, with streaks down the inner walls. Its
concentration is low (33rd percentile), spread along the whole rim. A sharp-rimmed impact crater with
gullied walls, a shadowed wall and a bright floor is a high-relief, high-frequency structure that a
256-float global code can only sketch. Such large fresh craters are uncommon at this crop scale,
where most crops are plains, dunes and knobby terrain. The pattern is physically coherent (the error
sits exactly where relief is steepest) and does *not* resemble any of the injection signatures.

**#3 sample_05849 (SRC_008, 25° N): crater or pit with a fully black shadow. Hypothesis: genuine
steep-walled depression at the limit of the sensor's dynamic range.** 11 % of the pixels are clipped
at 0 (saturation at the 100th percentile). The error traces the **rim and the sunlit wall**, where
the illuminated rock meets the clipped shadow. The shadow sits on the wall facing away from the sun,
consistent with the illumination of the surrounding small craters, which argues for real topography
rather than a synthetic black blob. An exposure set for the bright plain cannot record the shadowed
interior, so it clips. This is either a very deep fresh crater or a collapse pit. It is rare, but
Martian.

**#4 sample_03423 (SRC_128, 90° N): polar layered terrain with a large filled no-data area.
Hypothesis: genuine polar geology plus the residual no-data effect.** The valid area is a rotated
strip, with the two wedges masked and filled. It shows a bright, smooth upper surface and a darker
lower surface separated by a curved scarp with fine parallel banding (periodicity at the 98th
percentile). At 90° N this is the morphology of the **north polar layered deposits**, stacked ice and
dust layers exposed on a scarp. The reconstruction error is the *lowest* of the top 5, so again the
novelty is configurational: rare polar terrain, and an unusually large flat fill region, as
discussed in the v3/v4 changelog.

**#5 sample_03383 (SRC_067, 15° S): dark, low-sun rugged relief. Hypothesis: genuine massifs or
ridges under oblique illumination.** The frame is darker than 99 % of reference crops, and its
contrast is at the 98th percentile. The error is concentrated on the **thin sunlit crests**, not in
the shadows. Long shadows from steep spurs make a high-contrast, high-frequency pattern whose exact
crest geometry the code cannot store. SRC_067 is one of the four significantly enriched sources
(7 flags in 92 crops), which is what we would expect if the whole observation images unusual rugged
terrain.

**Beyond the top 5 (Fig 3.1b, diagnostic).** The heatmap *pattern* separates the injected content
from the rare geology above:

* The **Pentagon** aerial photograph (rank 47) and the **periodic grid** (rank 11), which on close
  inspection resembles an orchard (rows of trees seen from above), have the two highest
  reconstruction errors in the whole dataset (100th percentile). Their error is **spread
  diffusely over the entire frame** (concentration 0.18 and 0.16, i.e. almost uniform). The decoder
  has no Martian vocabulary for rectilinear buildings or a perfect lattice, so it fails everywhere.
  This is the textbook **non-Martian domain-shift** signature.
* The **black bar** and **two streaks**, which are *not* flagged by the latent score (ranks 1,120 and
  2,211), nevertheless sit in the **top 1 % of reconstruction error**, with the error lying *along the
  straight dark defect* (line scores 4.6 and 5.4). This is the **synthetic sensor-artefact**
  signature, and it shows the global latent's blind spot for small local defects (changelog, v3
  and v4).

| heatmap pattern | examples | physical hypothesis |
|---|---|---|
| ring following a rim, error on steep walls | 06917, 05849 | real high-relief feature beyond the code's resolution (rare geology) |
| error on thin bright crests of a dark frame | 03383 | oblique illumination of rugged relief |
| low error, high novelty | 00849, 03423 | rare *configuration*: straight seam (splice / CCD seam) or polar terrain + large fill |
| intense error spread over the whole frame | Pentagon 09828, grid 06476 | domain shift: non-Martian content |
| error along a straight dark line | bar 00015, streaks 00503 | synthetic sensor artefact (missed by the latent) |
"""

T["final"] = r"""
<div style="background:#fff6d6;border:2px solid #d9a400;border-radius:6px;padding:10px 16px">

**FINAL ANSWERS**

| | |
|---|---|
| **Final model** | **v3**: convolutional autoencoder trained from scratch; global linear bottleneck → fixed-length **256-d 1-D latent**; loss 0.84·(1−SSIM) + 0.16·L1 + 0.1·Gradient-L1 on valid pixels; map-projection no-data wedges masked |
| **Novelty Score** | Isolation Forest s(x, n) on the latents (500 trees × 5 seeds); **higher = more anomalous** |
| **Statistical threshold** | robust modified z-score > 3.5 (median + 3.5·MAD/0.6745) = **0.5293** |
| **Crops flagged** | **86 of 10,422 (0.83 %)**, listed in `outputs/flagged_anomalies.csv` |
| **Top-5 (Phase 3)** | sample_00849, sample_06917, sample_05849, sample_03423, sample_03383 |

</div>

**Key findings**

1. The decisive fix was **not** the loss but the data. Masking the map-projection no-data wedges
   (v3) removed a nuisance mode that had made every v1/v2 flag a border crop. It turned a bimodal
   score distribution into a unimodal one with a genuine tail.
2. The flagged set contains **clear non-Martian content** (an aerial photograph of the Pentagon,
   perfectly periodic grids), recognisable in Phase 3 by diffuse whole-frame reconstruction error.
   It also contains **rare genuine terrain** (north-polar layered deposits, fresh deep craters),
   which the metadata join (2.3) identifies as a terrain class concentrated in two polar
   observations.
3. The spatial bottleneck (v4) was tried and **rejected**, because its coordinates encoded local
   brightness.
4. **Limitations:** small local sensor defects are missed by latent-space scoring (the residual
   catches two of three), and recall for terrestrial content is incomplete (sample_05508).
"""
