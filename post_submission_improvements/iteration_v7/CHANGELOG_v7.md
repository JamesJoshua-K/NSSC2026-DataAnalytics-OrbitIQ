# Post-submission follow-up: v7 (not part of the Round 1 submission)

Work done after the 10 September deadline, for the Round 2 presentation. The submitted repository
is unchanged. Everything here lives in `post_submission_improvements/iteration_v7/`.

## v7: a second residual code for the *magnitude* of the failure

**Symptom.** In the submitted benchmark, raw reconstruction error (0.835 mean AUROC) separated the
planted fakes better than v6's latent (0.775). The evidence existed, but the latent was not
carrying all of it. Scoring the 64-d residual code on its own (0.790) already beat the full
320-d v6 latent. So the 256 scene dimensions were diluting the most useful part.

**Diagnosis.** v6's stage 2 encodes the *signed* residual, x minus x_hat. For ordinary terrain
that residual is a zero-mean texture of positive and negative ripples, and a 64-number code of it
keeps the *shape* of what stage 1 got wrong but only weakly keeps *how badly* it got it wrong.

**Fix.** Keep v3 (stage 1) and v6 (stage 2a) exactly as submitted. Add one new 64-d autoencoder,
stage 2b, trained on the absolute residual |x - x_hat| (same architecture, loss and training as
stage 2a). The Isolation Forest scores [2a | 2b], 128 numbers. Threshold rule unchanged.

**Why both codes.** Each residual code alone is weaker than the pair: signed 0.788, magnitude
0.823, pair 0.856 (dev), while doubling either one to 128 dimensions does not help (signed-128
0.776, magnitude-128 0.808). The two carry different information: shape and severity.

## Tried and rejected on the way

| Idea | Result (dev mean AUROC) | Why it failed |
|---|---|---|
| Stage 2 on the log error-energy heatmap | 0.761 | the log scale flattens exactly the whole-frame failures that identify Earth images |
| Smaller stage-1 code (128 / 64 / 32) | 0.69 / 0.57 / 0.54 (scene code) | a tight bottleneck organises the code around brightness; residuals fill with ordinary texture |
| Wider stage 1 (2x channels) | no gain over the same training at 1x | capacity was not the limit |
| Stage-2 gain 2 or 8, 16-256 dims, 60-100 epochs | all within 0.77-0.80 (signed) | the signed code saturates regardless |
| Train stage 1 for 100 / 200 epochs | looked like 0.906 / 0.926 | **a benchmark artefact**, see below |

## We found a flaw in our own benchmark

Longer training looked like a large win. But the 100-epoch stage 1 had a *worse* held-out
reconstruction error than v3 and a *better* error on its training crops: it was memorising. Our
benchmark compared unseen probes against real crops the network had trained on, so a memorising
model could separate *seen from unseen* without detecting anything. Direct check: its scores
separate unseen clean crops from trained ones with AUROC 0.593, where v6 scores 0.478 (chance).

In the real task the organisers' fakes are *inside* the training data, so we built a faithful
benchmark: inject 240 fakes (2.3%) into the dataset, replacing the crop each was made from,
retrain the whole pipeline, and score the injected crops against the rest. On it, training length
makes no difference (30 / 60 / 100 epochs: 0.822 / 0.818 / 0.813), so we kept the submitted stage 1.

## Results

Faithful injection benchmark, known fake types, two injection seeds, pipeline retrained each time:

| | Splice | Glitch | Domain | Mean | Recall at threshold | Precision |
|---|---|---|---|---|---|---|
| v6 recipe | 0.678 | 0.663 | 0.907 | 0.728 | 10% | 39% |
| v7 recipe | 0.766 | 0.829 | 0.863 | **0.822** | 26% | 56% |

## Final tests (selection rule fixed in PREREGISTRATION.md first; each scored once)

| Test | v6 submitted | **v7** | v7 + scene code |
|---|---|---|---|
| T1: 14 fake types, 300 each (6 types never used in development) | 0.742 | **0.828** | 0.820 |
| T1, known types only / never-seen types only | 0.770 / 0.706 | **0.853 / 0.794** | 0.836 / 0.800 |
| T2: fresh injection into training, known types | 0.732 | **0.800** | 0.817 |
| T2: fresh injection into training, 6 never-seen types | 0.631 | **0.732** | 0.742 |

v7 improves 12 of the 14 fake types in T1. It is worse on regular man-made textures: orchard-like
grids 0.884 -> 0.787 and field patchworks 0.957 -> 0.831. Keeping the scene code recovers the
patchworks (0.939), so the scene code still carries the "this texture is not Martian" signal.
Seen-vs-unseen check for v7: 0.498 (no memorisation advantage; v6 0.478).

## On the organisers' real data

Same Isolation Forest settings, same k = 3.5 rule: threshold 0.5105, **56 crops flagged** (v6: 48).
The four known Earth images rank 1, 2, 3 and 7. Crops v6 missed that v7 now flags, all visibly
tampered (figure `figs/v7_found_real.png`):

| Crop | What it is | v6 rank | v7 rank |
|---|---|---|---|
| sample_09747 | pasted rectangular patch | 72 | 9 |
| sample_00316 | pasted rectangular patch | 482 | 25 |
| sample_01815 | pasted rectangular patch | 680 | 38 |
| sample_00015 | synthetic black bar | 344 | 20 |
| sample_00043 | horizontal striping | 1,219 | 28 |
| sample_06857 | Earth: aerial view of buildings (not spotted before) | 205 | 39 |

What v7 stopped flagging is mostly genuine rare terrain, such as deep craters. The two polar
photographs that supplied 12 of v6's 48 flags now supply 2 of v7's 56 (v7 still flags 13 crops above
80 degrees latitude, spread over other photographs; flags now come from 37 source images, not 30). The two-streaks crop improves (2,517 -> 1,554) but is still
missed, and the dead-pixel-blocks crop gets worse (479 -> 8,951): small, isolated defects remain the
weak spot.

## Limits to state before a judge does
- The planted fakes are ours; the organisers' may differ. The real-data crops above are the
  closest thing to ground truth we have, and they point the same way.
- v7 trades some sensitivity to regular man-made textures for much better splice and glitch
  detection; adding the scene code back is the balanced variant.
- Small isolated defects (dead-pixel blocks, thin streaks) are still averaged away.
