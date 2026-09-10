# Engineering Changelog - Mars HiRISE Unsupervised Anomaly Detection

Every version uses identical data (10,422 crops), a random 90/10 split held out for loss curves,
AdamW + one-cycle LR (max 1e-3), batch 64, 30 epochs, horizontal/vertical flip augmentation,
bf16 autocast, seed 42, and the same Isolation Forest (500 trees, ψ = 256, 5 seeds averaged).
Each version changes **one** thing, so its effect can be attributed.

---

## v1 - Baseline: convolutional AE, pure MSE

**Setup.** 5× stride-2 conv encoder (32-64-128-256-256) → Linear → 256-d latent; mirrored
transposed-conv decoder; loss = MSE.

**Result.** Held-out MSE 0.00189, SSIM 0.705 after 30 epochs (plateaued from epoch ~20).

### Symptom 1 - Blurry reconstructions, fine texture erased
Dune ripple trains, small-crater fields and rough ground are reconstructed as smooth grey gradients
(outputs/v1/recon_grid.png, random crops 07869 and 01503). The error maps of *normal* crops are
dominated by texture, not by anything anomalous.

### Symptom 2 - The novelty engine ranks map-projection borders, not anomalies
* All of the top-25 Novelty Scores are crops with a black no-data wedge (the edge of the
  map-projected HiRISE strip). Their error maps are just a bright line tracing the wedge edge.
* Spearman ρ(novelty, reconstruction MSE) = −0.095: what the forest isolates is unrelated to what
  the decoder fails to explain.
* The score histogram has a long secondary shoulder (0.45-0.63) holding about 15 % of the crops, which
  is roughly the share of crops with a border. Even the robust modified-z rule (median + 3.5 MAD) therefore
  flags 1,596 crops (15.3 %), and the elbow and GMM rules flag even more. No defensible threshold can
  exist while a nuisance factor forms its own mode.
* One latent coordinate has variance 2,800 against a median of 33, so the code spends its largest
  axis on border geometry.

### Diagnosis
1. **Blur:** MSE is minimised by the conditional mean. When the exact phase of a ripple or
   the rim of a small crater is uncertain, averaging the possibilities gives the lowest expected
   squared error. The decoder learns to hedge, which erases exactly the high-frequency geology we
   need in order to notice unfamiliar texture.
2. **Borders:** a no-data wedge is a 0-valued region with a razor-sharp straight edge at an
   arbitrary angle and position. It is the highest-contrast structure in the dataset. Under MSE it
   dominates the loss, so the encoder dedicates capacity to encoding wedge geometry. In the latent
   space, crops with rare wedge angles are then the easiest points to isolate.

### Fix
Addressed one at a time: v2 changes only the loss (symptom 1); v3 addresses the no-data
nuisance (symptom 2).

---

## v2 - Structural loss: 0.84·(1 − SSIM) + 0.16·L1 + 0.1·Gradient-L1

**Change (loss only; architecture, data and schedule identical to v1).**

* **SSIM** (11×11 Gaussian window) penalises lost local contrast and structure even when mean
  brightness is right, which directly targets blur. The 0.84 / 0.16 split follows Zhao et al. (2017,
  *Loss Functions for Image Restoration with Neural Networks*).
* **L1** replaces MSE as the intensity anchor. SSIM ignores uniform offsets, and L1 is less dominated
  by a few huge errors than MSE.
* **Gradient-L1** on finite differences penalises smeared ridgelines and crater rims.

**Outcome.**

| | v1 | v2 |
|---|---|---|
| held-out SSIM | 0.7053 | 0.7065 |
| held-out MSE | 0.00189 | 0.00228 |
| PCA components for 90 % latent variance | 44 | 36 |
| Spearman ρ(novelty, no-data fraction) | 0.63 | **0.66** |
| crops flagged by modified-z > 3.5 | 1,596 | 1,985 |

### Symptom 1 (blur) - essentially unchanged
Crater rims are slightly crisper, but dune ripples (crop 01503) and small-crater fields are still
reconstructed as smooth gradients, and held-out SSIM moved by only +0.001.

**Diagnosis.** The loss was not the binding constraint. The code has only 256 floats for 51,529
pixels (≈200× compression), squeezed through a single linear layer from an 8×8×256 map. The exact
phase of a ripple field is high-entropy information that such a code cannot store. Whatever the
loss, the decoder can only output the "texture class" with its details averaged out. **Lesson:**
the loss decides *how* the unavoidable error is distributed; it cannot create information the
bottleneck discarded. For anomaly detection this matters less than it first seemed. The latent
only has to encode *what kind of* terrain a crop is, which it can do without reproducing every
ripple.

### Symptom 2 (no-data borders) - got worse
The novelty histogram (outputs/v2/scores.png) is now clearly **bimodal**. The second mode
(0.47-0.60) is almost entirely crops with a black no-data wedge. ρ(novelty, no-data fraction), already
0.63 under v1, rose to 0.66. The number flagged grew from 1,596 to 1,985, and every crop in the
top 25 has a wedge.

**Diagnosis.** Both new loss terms are *edge-sensitive*. The wedge boundary is a 0 → bright step,
the strongest gradient in the dataset, so SSIM and gradient-L1 reward the network for placing it
exactly. The encoder therefore spends *more* of its (already scarce) capacity on wedge angle and
position (the latent spectrum became more concentrated, 44 → 36 PCA components). In latent space,
crops with unusual wedge geometry are the easiest to isolate. The wedge is an artefact of
map-projecting the HiRISE strip, not terrain, so every one of those flags is a false positive by
construction.

### Fix → v3
Stop showing the network the wedge. Detect no-data regions, fill them with the crop's median, and
exclude them from the loss and error maps, keeping the v2 loss.

---

## v3 - No-data masking (data/loss change; architecture and loss weights as v2)

**Change.** A no-data mask per crop marks near-black (≤ 5) regions that survive a 5×5 opening,
touch the crop border and cover at least 0.5 % of it, dilated by 2 px for JPEG ringing. The opening
matters: thin interior dark lines, which could themselves be sensor-glitch anomalies, are
deliberately **not** masked (outputs/nodata_check.png). Masked pixels are filled with the crop's
valid-pixel median and excluded from every loss term, metric and error map. 19.7 % of crops have a
mask, covering 18.8 % of those crops on average.

**Outcome.**

| | v2 | v3 |
|---|---|---|
| held-out SSIM (v3: valid pixels) | 0.7065 | 0.7195 |
| Spearman ρ(novelty, no-data fraction) | 0.66 | **0.11** |
| Spearman ρ(novelty, reconstruction error) | 0.08 | **0.44** |
| PCA components for 90 % latent variance | 36 | **190** |
| score distribution | bimodal | unimodal, right-skewed tail |
| flagged by modified-z > 3.5 | 1,985 (19 %) | **86 (0.83 %)** |
| μ + 3σ (cross-check) | 45 | 95 |

* The nuisance mode is gone. With no second mode, the robust modified-z rule and μ + 3σ now roughly
  agree (86 vs 95 flagged), which suggests the tail is real rather than an artefact of the rule.
* Novelty now correlates with what the decoder fails to explain (ρ 0.08 → 0.44), and the code uses
  far more of its dimensions (190 components for 90 % variance instead of 36).
* The flagged set now contains clear content anomalies: an **aerial photograph of the Pentagon**
  (crop 09828), two **perfectly periodic grid textures** (06476, 07971) and striped strips (08731,
  06424) (outputs/v3_flagged_grid.png). Under v1 and v2 these were buried beneath the border crops:

  | spot-check crop | v1 rank | v2 rank | v3 rank |
  |---|---|---|---|
  | 06476 periodic grid | 10,225 | 5,839 | **11** |
  | 07971 periodic grid | 7,163 | 3,281 | **21** |
  | 09828 Pentagon (terrestrial) | 2,875 | 2,275 | **47** |
  | 05395 bright terrain + thin dark line | 784 | 1,735 | **12** |

### Symptom - small local glitches are ranked far below the threshold
During the sanity pass we noticed crops with obvious synthetic defects: a thick black bar
(00015), two black streaks (00503) and square dead-pixel blocks (00676). These are qualitative
spot checks we saw by eye; we used them only to diagnose, never to tune the threshold. v3 ranks them
1,120th, 2,211th and 304th, far below the 86 flagged, even though their *reconstruction error* is
visibly concentrated on the defect. A secondary residual: 41 of the 86 flagged crops still have a
(filled) wedge, against a 20 % base rate. A large perfectly flat region is itself uncommon.

### Diagnosis
The bottleneck is one **global** linear layer from the whole 8×8×256 feature map to 256 numbers,
and that layer is trained to summarise normal terrain. A defect covering about 2 % of the pixels
barely moves this global summary: the encoder projects the crop onto the normal-terrain manifold
and "forgets" the bar. The decoder then reproduces clean terrain, which is why the defect shows up
in the residual but not in the latent. Latent-density detection with a global code is sensitive to
*scene-level* novelty (Earth imagery, alien textures) and blind to *local* defects (glitches,
splice seams).

### Fix → v4
Keep the information local. Replace the global linear bottleneck with a **spatial bottleneck**,
a 1×1 convolution to an 8×8 grid of 4 channels, flattened to the same 256-d 1-D vector. Each
coordinate then describes one ~63 px neighbourhood, so a local defect drives a few coordinates to
extreme values, which is exactly what an Isolation Forest's axis-aligned random splits isolate fastest.

---

## v4 - Spatial bottleneck (architecture change; loss and masking as v3) - REJECTED

**Change.** The encoder's 8×8×256 map is reduced by a 1×1 convolution to 8×8×4 and flattened to a
256-d 1-D latent. The latent size, compression ratio, loss, masking and schedule are all as in v3.
Dropping the two 16,384×256 linear layers cuts the model from 11.1 M to 2.7 M parameters.

**Outcome - the hypothesis failed.**

| | v3 | v4 |
|---|---|---|
| held-out SSIM (valid pixels) | 0.7195 | 0.7159 |
| flagged by modified-z > 3.5 | 86 | 510 |
| Spearman ρ(novelty, reconstruction error) | 0.44 | 0.22 |
| Spearman ρ(novelty, deviation from typical brightness) | 0.36 | **0.66** |
| share of flagged crops among the 5 % darkest + 5 % brightest | 58 % | **97 %** |
| rank of black bar / two streaks / dead-pixel blocks | 1,120 / 2,211 / 304 | 2,833 / 3,158 / 872 |
| rank of Pentagon / periodic grids | 47 / 11, 21 | 488 / 178, 815 |

### Symptom
The local glitches did **not** move up; they moved *down*. The content anomalies that v3 caught
(the Pentagon, the periodic grids) were pushed out of the flagged set, and the flagged set grew
six-fold. It is now almost entirely very dark or very bright scenes
(outputs/v4_flagged_grid.png), which is exactly the "brightness heuristic" failure the organisers
warned about.

### Diagnosis
With only 4 channels per cell and no mixing across cells, each coordinate becomes a *local
albedo / local gradient* descriptor. An Isolation Forest isolates points that are extreme along
individual coordinates, and in a code made of local-albedo features the most extreme crops are simply
the darkest and brightest scenes. The injected defects are anomalous in their **configuration** (a
perfectly straight dark bar, square blocks, a regular grid), not in their local brightness. At the
scale of a 63 px cell, a black bar is about as dark as an ordinary shadow, so none of its coordinates
is extreme. The global code of v3 can represent "this scene contains an unnatural structure",
because its linear layer combines evidence across the whole crop. The per-cell code cannot.

### Decision
**Revert to v3 as the final model.** v4 stays in the journal as a documented negative result. The
choice is made on label-free evidence: v3's novelty agrees better with what the decoder fails to
explain (ρ 0.44 vs 0.22), it depends much less on raw brightness, and its score distribution gives
a small tail on which the robust and Gaussian rules agree (86 vs 95). The spot checks only
corroborate this.

**Remaining limitation (honest).** Recall is incomplete. A second Earth image (an aerial view of
buildings, crop 05508, spotted in v4's grid) ranks 354th under v3 and is not flagged. Small
local defects are still missed by *latent-space* scoring in every version. The decoder residual
catches two of the three: the black bar and the two streaks are in the **top 1 % of reconstruction
error**, with their error lying along the defect (Fig 3.1b). The dead-pixel blocks are invisible even
there (53rd percentile). Residual-based or patch-level scoring is therefore a promising extension,
but not a complete answer. v3 also keeps some brightness bias:
58 % of its flags lie in the brightness extremes, against a 10 % base rate. Some of that is
genuine, because unusually bright or dark terrain (frost, deep shadow) *is* rarer.

---
