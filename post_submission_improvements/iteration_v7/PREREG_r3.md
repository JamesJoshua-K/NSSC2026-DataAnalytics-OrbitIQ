# Round-3 search ("ultimate model"): rules fixed before any result (2026-10-01)

Post-submission work only. Baseline to beat: v7 recipe = [signed | abs] stage-2 codes on the submitted
30-epoch stage 1, Isolation Forest 500 trees, max_samples 256, 5 seeds, modified z > 3.5.

## Development (selection) data
Injection benchmark (fakes injected into the training data, whole pipeline retrained, `p3.py`):
- seed 100 and seed 101: the 8 known generator types (benchmark.py)
- seed 102: the 6 held-out generator types (r2lib.NEW_GENERATORS), already seen once during v7 work
Every candidate runs on all three. All candidates share the same injected crops per seed (paired comparison).

## Selection rule
dev score = mean over the three runs of each run's mean per-type AUROC.
A candidate replaces v7 only if
1. its dev score beats the v7 recipe by more than 0.015 (about 1.5x the observed seed-to-seed spread of
   paired differences), AND
2. it is not worse than v7 on any of the three runs by more than 0.02, AND
3. all four real Earth crops (orchard 06476, grid 07971, runway 07701, Pentagon 09828) stay in its top 25 on
   every run (they are genuine planted fakes in the organisers' data and are never overwritten by injection).
Ties within 0.015 go to the simpler model (fewer networks, fewer settings).
Offline Isolation-Forest settings and latent compositions are candidates like any other and face the same rule.

## Final test (run once, after selection, for the selected model and the v7 recipe only)
Fresh injections never used before:
- seed 300: the 8 known types
- seed 301: the 6 held-out types
- seed 302: a BLIND set of new generator types written by an independent cloud session that never saw any
  tuning result (if that set is not available, seed 302 is dropped and this file is amended before testing).
Reported: per-type AUROC and means, recall and precision at the threshold, Earth ranks.
Plus the real competition data (no injection): number flagged, ranks of the four Earth images and of the six
tampered crops found by v7 (09747, 00316, 01815, 00015, 00043, 06857).
