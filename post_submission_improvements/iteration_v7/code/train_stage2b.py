"""Retrain v7's one new network, stage 2b, from the submitted stage 1. Nothing else is trained.

Stage 2b is the submitted stage-2 design (64-number code) trained on the magnitude of stage 1's
residual, clip(4 |x - x_hat|, 0, 1), with the submitted recipe: MSE + 0.5 L1 on valid pixels,
30 epochs, AdamW, one-cycle LR (peak 1e-3), batch 64, flips, 10% held out, seed 42.
GPU training is not bit-for-bit deterministic, so the result differs slightly from weights/stage2b_abs64.pt.

usage: python post_submission_improvements/iteration_v7/code/train_stage2b.py [--out PATH]
"""
import argparse, os
import torch
from r2lib import *

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(R2, "weights", "stage2b_retrained.pt"))
args = ap.parse_args()

X, M = load_data()
Xin = fill_nodata(X, M)
s1 = load_model("v3", os.path.join(ROOT, "outputs"))      # submitted stage 1, frozen
A = stage2_images(s1, Xin, M, "abs")                       # magnitude pictures, uint8

seed_everything(42)
s2b = ConvAE(latent_dim=64)
s2b, hist, _ = train_autoencoder(s2b, A, ReconLoss(w_mse=1.0, w_l1=0.5), epochs=30, lr=1e-3, masks_u8=M, seed=42)
torch.save(s2b.state_dict(), args.out)
hist.to_csv(os.path.splitext(args.out)[0] + "_history.csv", index=False)
print("saved", args.out)
