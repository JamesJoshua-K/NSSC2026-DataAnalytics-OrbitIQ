"""Spot-check diagnostics across versions: where do visually noticed crops rank, and what does a
version flag? The spot-check crops were noticed by eye during the sanity pass / earlier versions;
they are used only to diagnose failure modes, never to tune the threshold.

usage: python inspect_version.py v4
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V = sys.argv[1]
X = np.load("cache/images_u8.npy")
nd = (X <= 2).mean((1, 2))
SPOT = {15: "black bar", 503: "two streaks", 676: "dead-pixel blocks", 9828: "Pentagon (terrestrial)",
        5508: "terrestrial buildings", 6476: "periodic grid", 7971: "periodic grid", 5395: "bright + thin line",
        11: "dark terrain (control)", 385: "dark terrain (control)"}


def mad_thr(n):
    med = np.median(n)
    return med + 3.5 * np.median(np.abs(n - med)) / 0.6745


rows = []
for v in ["v1", "v2", "v3", "v4"]:
    try:
        n = np.load(f"outputs/{v}/novelty.npy")
    except FileNotFoundError:
        continue
    rank = (-n).argsort().argsort() + 1
    thr = mad_thr(n)
    for k, lab in SPOT.items():
        rows.append(dict(version=v, crop=f"{k:05d}", note=lab, rank=int(rank[k - 1]), flagged=bool(n[k - 1] > thr)))
    f = n > thr
    print(f"{v}: flagged {int(f.sum())} | with no-data wedge {int((nd[f] > .01).sum())} | threshold {thr:.4f}")
t = pd.DataFrame(rows)
t.to_csv("outputs/spotcheck_ranks.csv", index=False)
print(t.pivot(index=["crop", "note"], columns="version", values="rank").to_string())
print(t.pivot(index=["crop", "note"], columns="version", values="flagged").to_string())

n = np.load(f"outputs/{V}/novelty.npy")
thr = mad_thr(n)
o = np.argsort(-n)
nf = int((n > thr).sum())
sel = list(o[:min(48, nf)]) + list(o[max(nf - 4, 0):nf + 4])
cols = 8
rows_ = int(np.ceil(len(sel) / cols))
fig, ax = plt.subplots(rows_, cols, figsize=(16, 2.15 * rows_))
for a in ax.ravel():
    a.axis("off")
for j, (a, i) in enumerate(zip(ax.ravel(), sel)):
    r = int((-n).argsort().argsort()[i]) + 1
    a.imshow(X[i], cmap="gray", vmin=0, vmax=255)
    a.set_title(f"#{r} {i + 1:05d} {n[i]:.3f}" + (" W" if nd[i] > .01 else ""), fontsize=8,
                color="k" if r <= nf else "b")
plt.suptitle(f"{V}: top flagged (black) and ranks around the threshold (blue, not flagged); W = no-data wedge")
plt.tight_layout()
plt.savefig(f"outputs/{V}_flagged_grid.png", dpi=55)
print("grid ->", f"outputs/{V}_flagged_grid.png")
