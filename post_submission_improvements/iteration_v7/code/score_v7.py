"""Reproduce v7's result on the competition data from the saved weights (no training).

v7 = submitted stage 1 (outputs/v3) + submitted stage 2a (outputs/v6, signed residual)
   + the one new network, stage 2b (weights/stage2b_abs64.pt, magnitude of the residual).
The Isolation Forest scores [2a | 2b] (64 + 64 = 128 numbers); the threshold is the submitted
modified z-score rule, M > 3.5. Expected: threshold 0.5105, 56 crops flagged.

usage: python post_submission_improvements/iteration_v7/code/score_v7.py
"""
import os
import numpy as np, pandas as pd, torch
from r2lib import *

X, M = load_data()
Xin = fill_nodata(X, M)
out = os.path.join(ROOT, "outputs")
s1, s2a = load_model("v3", out), load_model("v6", out)
s2b = ConvAE(latent_dim=64)
s2b.load_state_dict(torch.load(os.path.join(R2, "weights", "stage2b_abs64.pt"), map_location=DEVICE))
s2b = s2b.to(DEVICE).eval()

Z2a = embed_and_reconstruct(s2a, residual_images(s1, Xin, M), M)[0]          # identical to the submitted v6 residual code
Z2b = embed_and_reconstruct(s2b, stage2_images(s1, Xin, M, "abs"), M)[0]
nov, nov_sd = novelty_scores(np.hstack([Z2a, Z2b]))
thr = mad_threshold(nov, 3.5)

files = pd.read_csv(os.path.join(ROOT, "data", "crop_metadata_index.csv")).filename.values
v6 = np.load(os.path.join(out, "v6", "novelty.npy"))
df = pd.DataFrame(dict(filename=files, novelty_v7=nov, novelty_v7_std=nov_sd, rank_v7=(-nov).argsort().argsort() + 1,
                       flagged_v7=nov > thr, novelty_v6=v6, rank_v6=(-v6).argsort().argsort() + 1,
                       flagged_v6=v6 > mad_threshold(v6, 3.5))).sort_values("rank_v7")
f = os.path.join(R2, "results", "v7_scores_real_recomputed.csv")
df.to_csv(f, index=False)
print(f"v7: threshold {thr:.4f}, {int((nov > thr).sum())} of {len(nov)} crops flagged (v6: {int(df.flagged_v6.sum())})")
print("ranks of the four Earth images:",
      {k: int(df.set_index('filename').rank_v7[f'sample_{k}.jpg']) for k in ("06476", "07971", "07701", "09828")})
print("written", f)
