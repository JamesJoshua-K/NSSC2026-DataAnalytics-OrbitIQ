# Round-2 follow-up: selection rule fixed before the held-out test (2026-09-29)

All selection so far used ONLY the dev benchmark (the submitted planted-fakes benchmark, seed 0,
8 generator types). The test benchmark has not been scored for any candidate.

Test benchmark: 300 fresh probes (seed 1) of each of the same 8 types, plus 300 probes (seed 2)
of each of 6 generator types never used during development: splice_matched, splice_feather,
glitch_noise, glitch_saturation, domain_pivots, domain_fields.

Unchanged from the submission: Isolation Forest (500 trees, psi 256, 5 seeds, fitted on the
10,422 real crops only), modified z-score threshold k = 3.5, no-data masking, stage-1 loss.

## Candidates
- **Baseline**: v6 exactly as submitted (checkpoint reproduces the submitted latents, max diff 0.0).
- **v7 (headline, chosen on dev)**: stage 1 = v3 architecture and loss trained 100 epochs instead
  of 30; stage 2a = 64-d autoencoder of the signed residual (as v6); stage 2b = 64-d autoencoder of
  the absolute residual |x - x_hat|; Isolation Forest on [2a | 2b] (128-d).
- **v7 + scene**: same, with the stage-1 scene code PCA-reduced to 64-d prepended (equal share per
  code, 192-d). Reported alongside, not selected.

Pending dev runs that may change the stage-1 recipe: E2 (200 epochs), E4 (100 epochs, seed 1),
E5 (512-d, 100 epochs). Rule: switch to 200 epochs or 512-d only if dev mean AUROC improves by
more than 0.01 (the observed seed noise is about 0.004). E4 is a robustness check, not a candidate.

## Reported on test, once
Mean AUROC (all 14 types), mean on the 8 known types, mean on the 6 held-out types, per class;
recall at the statistical threshold; plus real-data checks (number flagged, ranks of the four
Earth images, the black bar and two streaks crops).

## Amendment (2026-09-29, before any final test is scored)
The dev benchmark was found to reward memorisation: probes are unseen images while the negatives
were seen in training. A 100-epoch stage 1 separates unseen clean crops from trained ones
(AUROC 0.593; submitted v6: 0.478). A faithful benchmark was therefore built (`p2.py`): fakes are
injected INTO the dataset (2.3%, replacing their source crop), the whole pipeline is retrained, and
the injected crops are scored against the rest. On it, stage-1 training length makes no difference
(30 / 60 / 100 epochs: 0.822 / 0.818 / 0.813 for [signed|abs], two injection seeds), so the
100-epoch gain was entirely memorisation.

**Revised v7**: stage 1 = v3 exactly as submitted; stage 2a = v6 exactly as submitted (signed
residual); stage 2b = NEW 64-d autoencoder of the absolute residual (30 epochs, seed 42). Headline
latent [2a | 2b] (128-d); [PCA64(scene) | 2a | 2b] reported alongside. Only one new network.

Note: one P2 run used the held-out generator types (seed 102); it was viewed while establishing
the confound, and did not drive a choice. Final tests, run once, on data never scored before:
- T1: the original test probes (seed-1 known types + seed-2 new types), v6 submitted vs v7.
- T2: fresh injection sets with the 30-epoch recipe: known types seed 200, new types seed 201.
