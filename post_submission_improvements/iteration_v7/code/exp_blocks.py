"""Round-2 experiment 1: how much of v6's signal is lost by the Isolation Forest mixing
256 scene dims with 64 residual dims? Uses the submitted v3/v6 checkpoints unchanged."""
import os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))     # .../iteration_v7/code
R2 = os.path.dirname(HERE)                             # .../iteration_v7: weights, results, regenerated outputs
ROOT = os.path.dirname(os.path.dirname(R2))            # repository root: nssc_core, benchmark, outputs/, cache/
sys.path.insert(0, ROOT)
from nssc_core import *
from benchmark import make_probes, forest_scorer, summarise
from sklearn.decomposition import PCA

C = os.path.join(R2, "cache")
X = np.load(os.path.join(ROOT, "cache/images_u8.npy"))
probes, meta = make_probes(X, (X <= 2).mean((1, 2)), 300, seed=0)
f = os.path.join(C, "v6_blocks_dev.npz")
if not os.path.exists(f):
    out = os.path.join(ROOT, "outputs")
    Zr, mser, _ = version_latents("v6", X, out, mask_cache=os.path.join(ROOT, "cache/nodata_masks.npy"))
    Zp, msep, _ = version_latents("v6", probes, out)
    np.savez(f, Zr=Zr, Zp=Zp, mser=mser, msep=msep)
d = np.load(f)
Zr, Zp = d["Zr"], d["Zp"]
Z1r, Z2r, Z1p, Z2p = Zr[:, :256], Zr[:, 256:], Zp[:, :256], Zp[:, 256:]

def run(name, A, B):
    s = forest_scorer(A)
    rows, head = summarise(name, s(A), s(B), meta)
    det = pd.DataFrame(rows)
    cls = det.groupby("cls").auroc.mean()
    per = det.set_index("type").auroc
    print(f"{name:28s} mean {det.auroc.mean():.3f} | splice {cls['splice']:.3f} glitch {cls['glitch']:.3f} "
          f"domain {cls['domain']:.3f} | flags {head['n_flag_real']:4d} | Earth ranks "
          f"{head['rank_orchard grid']},{head['rank_grid']},{head['rank_Pentagon']} | "
          + " ".join(f"{v:.2f}" for v in per.values), flush=True)

print("per-type order:", list(pd.DataFrame(summarise('x', np.zeros(len(Zr)), np.zeros(len(Zp)), meta)[0]).type))
run("v6 as submitted [Z1|Z2]", Zr, Zp)
run("Z1 only (scene)", Z1r, Z1p)
run("Z2 only (residual)", Z2r, Z2p)
for k in (2, 4):
    run(f"[Z1 | Z2 x{k}]", np.hstack([Z1r] + [Z2r] * k), np.hstack([Z1p] + [Z2p] * k))
for k in (16, 32, 64):
    p = PCA(k, random_state=0).fit(Z1r)
    run(f"[PCA{k}(Z1) | Z2]", np.hstack([p.transform(Z1r), Z2r]), np.hstack([p.transform(Z1p), Z2p]))
