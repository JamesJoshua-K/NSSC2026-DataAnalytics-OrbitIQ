"""Slide figures for the post-submission v7 follow-up."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from r2lib import *

FIG = os.path.join(R2, "figs"); os.makedirs(FIG, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BEFORE, AFTER = "#86b6ef", "#184f95"          # one hue, two shades: before -> after
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})

# ---------------------------------------------------------------- 1. before -> after dumbbell
rows = [("Fair benchmark: fakes injected into training\n(known types, 2 retrained runs)", 0.728, 0.822),
        ("Final test T1: 14 fake types\n(6 never used during development)", 0.742, 0.828),
        ("Final test T2: fresh injection,\nknown types", 0.732, 0.800),
        ("Final test T2: fresh injection,\n6 never-seen types", 0.631, 0.732)]
fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURF); ax.set_facecolor(SURF)
for y, (lab, a, b) in enumerate(rows[::-1]):
    ax.plot([a, b], [y, y], color="#b9b8b2", lw=2, zorder=1, solid_capstyle="round")
    ax.scatter([a], [y], s=110, color=BEFORE, zorder=3, edgecolors=SURF, linewidths=2)
    ax.scatter([b], [y], s=110, color=AFTER, zorder=3, edgecolors=SURF, linewidths=2)
    ax.text(a - 0.008, y, f"{a:.3f}", ha="right", va="center", fontsize=10, color=INK2)
    ax.text(b + 0.008, y, f"{b:.3f}", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(0.985, y, f"+{b - a:.3f}", ha="right", va="center", fontsize=10.5, color=INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=10)
ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
ax.text(0.503, len(rows) - 0.45, "chance", fontsize=9, color=INK2, va="bottom")
ax.set_xlim(0.48, 1.0); ax.set_ylim(-0.6, len(rows) - 0.3)
ax.set_xlabel("mean AUROC (1.0 = every fake ranked above every real crop)")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.scatter([], [], s=90, color=BEFORE, label="v6 (submitted)"); ax.scatter([], [], s=90, color=AFTER, label="v7 (post-submission)")
ax.legend(loc="lower right", frameon=False, fontsize=10, bbox_to_anchor=(1.0, 1.0), ncol=2)
ax.set_title("v7 beats the submitted v6 on every test, including fake types it never saw",
             loc="left", fontsize=12.5, pad=24)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "v7_benchmarks.png"), dpi=200, facecolor=SURF); plt.close()

# ---------------------------------------------------------------- 2. what v7 caught in the real data
X, M = load_data()
s1 = load_model("v3", os.path.join(ROOT, "outputs"))
sc = pd.read_csv(os.path.join(R2, "results", "v7_scores_real.csv")).set_index("filename")
picks = [(9747, "pasted patch"), (316, "pasted patch"), (1815, "pasted patch"),
         (15, "synthetic black bar"), (43, "horizontal striping"), (6857, "Earth: aerial buildings")]
idx = np.array([k - 1 for k, _ in picks])
Xf = fill_nodata(X[idx], M[idx])
orig, rec = reconstruct(s1, Xf, np.arange(len(idx)))
fig, ax = plt.subplots(2, len(picks), figsize=(15, 5.6), facecolor=SURF)
for j, ((k, lab), i) in enumerate(zip(picks, idx)):
    r = sc.loc[f"sample_{k:05d}.jpg"]
    e = error_map(orig[j], rec[j], 2.0, valid=1 - M[i])
    ax[0, j].imshow(X[i], cmap="gray", vmin=0, vmax=255)
    ax[1, j].imshow(X[i], cmap="gray", vmin=0, vmax=255)
    ax[1, j].imshow(e, cmap="magma", alpha=0.6, vmin=0, vmax=np.percentile(e, 99.5))
    ax[0, j].set_title(f"{lab}\nsample_{k:05d}", fontsize=10)
    ax[1, j].set_xlabel(f"v6 rank {int(r.rank_v6)}  →  v7 rank {int(r.rank_v7)}", fontsize=10.5, color=INK)
    for a in ax[:, j]: a.set_xticks([]); a.set_yticks([])
ax[0, 0].set_ylabel("crop", fontsize=10); ax[1, 0].set_ylabel("stage-1 error map", fontsize=10)
fig.suptitle("Real dataset: crops v7 flags that v6 missed (v7 flags 56 crops; v6 flagged none of these)",
             fontsize=12.5, x=0.01, ha="left")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "v7_found_real.png"), dpi=170, facecolor=SURF); plt.close()
print("written:", os.listdir(FIG))
