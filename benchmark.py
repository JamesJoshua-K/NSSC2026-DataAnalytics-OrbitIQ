"""Planted-fakes benchmark for model selection.

Synthetic anomalies that mimic the three Genesis-Outlier classes (semantic splicing, sensor
glitches, non-Martian/man-made content) are injected into clean crops. Each version's latent +
Isolation Forest (fitted on the REAL 10,422 crops only) scores them. The competition threshold
rule (modified z > 3.5 on the real scores) is applied unchanged - the benchmark is used only to
COMPARE versions, never to tune the threshold or the anomaly count.

usage: python benchmark.py v1 v2 v3 v4 [--n 300] [--tag name]
"""
import argparse, os, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from scipy import ndimage
from nssc_core import *

ROOT = os.path.dirname(os.path.abspath(__file__))
S = IMG_SIZE
SPOT = {9828: "Pentagon", 6476: "orchard grid", 7971: "grid", 5508: "buildings",
        15: "black bar", 503: "two streaks", 676: "dead blocks"}


# ------------------------------------------------------------ anomaly generators (uint8 227x227)
def _bar_mask(rng, thick=(3, 8), length=(60, 160)):
    yy, xx = np.mgrid[0:S, 0:S]
    cx, cy = rng.uniform(40, S - 40, 2)
    ang, L, t = rng.uniform(0, np.pi), rng.uniform(*length), rng.uniform(*thick)
    along = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
    across = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
    return (np.abs(along) <= L / 2) & (np.abs(across) <= t / 2)


def splice_line(a, b, rng):
    """Two unrelated Mars crops joined along a straight seam at a random angle."""
    yy, xx = np.mgrid[0:S, 0:S]
    ang, off = rng.uniform(0, np.pi), rng.uniform(-40, 40)
    side = (xx - S / 2) * np.cos(ang) + (yy - S / 2) * np.sin(ang) > off
    out = a.copy(); out[side] = b[side]
    return out


def splice_patch(a, b, rng):
    """A rectangular patch of another Mars crop pasted in."""
    h, w = rng.integers(50, 120, 2)
    y, x = rng.integers(0, S - h), rng.integers(0, S - w)
    y2, x2 = rng.integers(0, S - h), rng.integers(0, S - w)
    out = a.copy(); out[y:y + h, x:x + w] = b[y2:y2 + h, x2:x2 + w]
    return out


def glitch_bars(a, b, rng):
    out = a.copy()
    for _ in range(rng.integers(1, 4)):
        out[_bar_mask(rng)] = rng.integers(0, 12)
    return out


def glitch_blocks(a, b, rng):
    out = a.copy()
    for _ in range(rng.integers(2, 7)):
        s = rng.integers(6, 18); y, x = rng.integers(0, S - s, 2)
        out[y:y + s, x:x + s] = rng.choice([0, 128, 255])
    return out


def glitch_stripes(a, b, rng):
    """Column-wise gain banding over a vertical band (CCD channel / read-out artefact)."""
    out = a.astype(float)
    x0, w, period = rng.integers(0, S - 60), rng.integers(40, 120), rng.integers(2, 6)
    cols = np.arange(x0, min(S, x0 + w))
    out[:, cols] *= (1 + rng.uniform(0.25, 0.5) * np.where((cols // period) % 2 == 0, 1, -1))[None, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def glitch_lines(a, b, rng):
    """Full-width line dropouts / hot lines."""
    out = a.copy()
    for _ in range(rng.integers(1, 5)):
        v, k, t = rng.choice([0, 255]), rng.integers(0, S - 3), rng.integers(1, 3)
        if rng.random() < .5: out[k:k + t, :] = v
        else: out[:, k:k + t] = v
    return out


def domain_grid(a, b, rng):
    """Man-made periodic texture (orchard / solar farm / field pattern), random orientation."""
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    ang, p = rng.uniform(0, np.pi), rng.uniform(8, 20)
    u = xx * np.cos(ang) + yy * np.sin(ang); v = -xx * np.sin(ang) + yy * np.cos(ang)
    blobs = np.cos(2 * np.pi * u / p) * np.cos(2 * np.pi * v / (p * rng.uniform(.8, 1.25)))
    out = rng.uniform(90, 170) + rng.uniform(30, 60) * blobs + rng.normal(0, 8, (S, S))
    return np.clip(out, 0, 255).astype(np.uint8)


def domain_urban(a, b, rng):
    """Rectilinear 'urban' scene: rectangles (buildings) and straight roads, randomly rotated."""
    out = np.full((S, S), rng.uniform(80, 150))
    for _ in range(rng.integers(15, 40)):
        h, w = rng.integers(6, 40, 2); y, x = rng.integers(0, S - h), rng.integers(0, S - w)
        out[y:y + h, x:x + w] = rng.uniform(30, 230)
    for _ in range(rng.integers(2, 6)):
        k, t, val = rng.integers(0, S - 8), rng.integers(3, 8), rng.uniform(40, 220)
        if rng.random() < .5: out[k:k + t, :] = val
        else: out[:, k:k + t] = val
    out = ndimage.rotate(out, rng.uniform(0, 90), reshape=False, mode="reflect")
    out += rng.normal(0, 6, (S, S))
    return np.clip(out, 0, 255).astype(np.uint8)


GENERATORS = {"splice_line": ("splice", splice_line), "splice_patch": ("splice", splice_patch),
              "glitch_bars": ("glitch", glitch_bars), "glitch_blocks": ("glitch", glitch_blocks),
              "glitch_stripes": ("glitch", glitch_stripes), "glitch_lines": ("glitch", glitch_lines),
              "domain_grid": ("domain", domain_grid), "domain_urban": ("domain", domain_urban)}


def make_probes(X, nodata_frac, n=300, seed=0):
    """n probes per generator, built from crops without no-data (fixed seed -> identical for every version)."""
    rng = np.random.default_rng(seed)
    clean = np.where(nodata_frac < 0.001)[0]
    src, partner = rng.choice(clean, n, replace=False), rng.choice(clean, n, replace=False)
    imgs, rows = [], []
    for name, (cls, fn) in GENERATORS.items():
        for i, j in zip(src, partner):
            imgs.append(fn(X[i], X[j], rng))
            rows.append(dict(type=name, cls=cls, source=int(i)))
    return np.stack(imgs), pd.DataFrame(rows)


# ------------------------------------------------------------ scoring
def forest_scorer(Z_real, seeds=(0, 1, 2, 3, 4)):
    forests = [IsolationForest(n_estimators=500, max_samples=256, random_state=s, n_jobs=-1).fit(Z_real) for s in seeds]
    return lambda A: np.mean([-f.score_samples(A) for f in forests], 0)


def summarise(name, nov_real, nov_probe, meta, recon_real=None, recon_probe=None):
    thr = mad_threshold(nov_real, 3.5)
    rows = []
    for t, g in meta.groupby("type", sort=False):
        p = nov_probe[g.index.values]
        y = np.r_[np.zeros(len(nov_real)), np.ones(len(p))]
        r = dict(model=name, type=t, cls=g.cls.iloc[0],
                 auroc=roc_auc_score(y, np.r_[nov_real, p]),                 # fake vs whole real dataset
                 recall=float((p > thr).mean()),                              # caught by the statistical threshold
                 median_pct=float(np.median(np.searchsorted(np.sort(nov_real), p) / len(nov_real) * 100)))
        if recon_real is not None:
            r["auroc_recon"] = roc_auc_score(y, np.r_[recon_real, recon_probe[g.index.values]])
        rows.append(r)
    rank = (-nov_real).argsort().argsort() + 1
    spot = {f"{lab}": int(rank[k - 1]) for k, lab in SPOT.items()}
    head = dict(model=name, n_flag_real=int((nov_real > thr).sum()), threshold=float(thr),
                spot_flagged=int(sum(nov_real[k - 1] > thr for k in SPOT)), **{f"rank_{k}": v for k, v in spot.items()})
    return rows, head


def eval_version(v, X, probes, meta):
    out = os.path.join(ROOT, "outputs")
    Z, mse, _ = version_latents(v, X, out, mask_cache=os.path.join(ROOT, "cache/nodata_masks.npy"))
    Zp, msep, _ = version_latents(v, probes, out)        # same pipeline a real crop would get
    score = forest_scorer(Z)
    return summarise(v, score(Z), score(Zp), meta, mse, msep)


def eval_raw_pixels(X, probes, meta, side=32):
    """Baseline: Isolation Forest on raw pixels downsampled to side x side (no learning at all)."""
    def feats(A):
        t = torch.from_numpy(A).float().unsqueeze(1).div(255)
        return F.adaptive_avg_pool2d(t, side).flatten(1).numpy()
    Zr, Zp = feats(X), feats(probes)
    score = forest_scorer(Zr)
    return summarise(f"raw pixels {side}x{side}", score(Zr), score(Zp), meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("versions", nargs="*")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--raw", action="store_true", help="include the raw-pixel baseline")
    ap.add_argument("--tag", default="benchmark")
    args = ap.parse_args()
    X = np.load(os.path.join(ROOT, "cache/images_u8.npy"))
    nd = (X <= 2).mean((1, 2))
    probes, meta = make_probes(X, nd, args.n)
    print(f"{len(probes)} probes ({args.n} per type x {len(GENERATORS)} types)")
    rows, heads = [], []
    jobs = [(v, lambda v=v: eval_version(v, X, probes, meta)) for v in args.versions]
    if args.raw:
        jobs.append(("raw", lambda: eval_raw_pixels(X, probes, meta)))
    for name, job in jobs:
        t0 = time.time(); r, h = job(); rows += r; heads.append(h)
        print(f"  {name}: done in {time.time() - t0:.0f}s")
    det = pd.DataFrame(rows); head = pd.DataFrame(heads)
    det.to_csv(os.path.join(ROOT, "outputs", f"{args.tag}_detail.csv"), index=False)
    by_cls = det.groupby(["model", "cls"], sort=False)[["auroc", "recall"]].mean().unstack("cls")
    by_cls.columns = [f"{m}_{c}" for m, c in by_cls.columns]
    summ = head.set_index("model").join(by_cls)
    summ["auroc_mean"] = det.groupby("model", sort=False).auroc.mean()
    summ["recall_mean"] = det.groupby("model", sort=False).recall.mean()
    if "auroc_recon" in det:
        summ["auroc_recon_mean"] = det.groupby("model", sort=False).auroc_recon.mean()
    summ.to_csv(os.path.join(ROOT, "outputs", f"{args.tag}_summary.csv"))
    pd.set_option("display.width", 250)
    print("\nPER TYPE (latent+IF AUROC vs whole real set | recall at the statistical threshold)")
    print(det.pivot_table(index="type", columns="model", values=["auroc", "recall"], sort=False).round(3).to_string())
    print("\nSUMMARY"); print(summ.round(3).T.to_string())


if __name__ == "__main__":
    main()
