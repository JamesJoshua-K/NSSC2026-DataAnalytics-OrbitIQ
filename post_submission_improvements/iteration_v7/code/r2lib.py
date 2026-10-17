"""Post-submission (v7) experiments. Builds on the submitted nssc_core / benchmark, unchanged.

Dev benchmark  = the submitted planted-fakes benchmark (seed 0, 8 generators) - used for selection.
Test benchmark = fresh probes (seed 1) of the same 8 generators  +  6 NEW generator types that are
                 never looked at during selection (seed 2). Reported once, at the end.
"""
import os, sys, json, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))     # .../iteration_v7/code
R2 = os.path.dirname(HERE)                             # .../iteration_v7: weights, results, regenerated outputs
ROOT = os.path.dirname(os.path.dirname(R2))            # repository root: nssc_core, benchmark, outputs/, cache/
sys.path.insert(0, ROOT)
from nssc_core import *
from nssc_core import _gauss_window
import benchmark as B
from sklearn.metrics import roc_auc_score

S = IMG_SIZE
CACHE = os.path.join(ROOT, "cache")


# ------------------------------------------------------------ held-out generator types (test only)
def splice_matched(a, b, rng):
    """Patch of another crop, brightness and contrast matched to the host: only texture differs."""
    h, w = rng.integers(60, 130, 2)
    y, x = rng.integers(0, S - h), rng.integers(0, S - w)
    y2, x2 = rng.integers(0, S - h), rng.integers(0, S - w)
    p, q = b[y2:y2 + h, x2:x2 + w].astype(float), a[y:y + h, x:x + w].astype(float)
    p = (p - p.mean()) / max(p.std(), 1) * q.std() + q.mean()
    out = a.copy(); out[y:y + h, x:x + w] = np.clip(p, 0, 255).astype(np.uint8)
    return out


def splice_feather(a, b, rng):
    """Two crops joined along a curved seam with a soft transition."""
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    ang, amp, per = rng.uniform(0, np.pi), rng.uniform(5, 25), rng.uniform(80, 200)
    u = (xx - S / 2) * np.cos(ang) + (yy - S / 2) * np.sin(ang)
    v = -(xx - S / 2) * np.sin(ang) + (yy - S / 2) * np.cos(ang)
    d = u - amp * np.sin(2 * np.pi * v / per) - rng.uniform(-30, 30)
    t = 1 / (1 + np.exp(-d / rng.uniform(2, 4)))
    return np.clip(a * (1 - t) + b * t, 0, 255).astype(np.uint8)


def glitch_noise(a, b, rng):
    """Burst of read-out noise over a rectangle."""
    out = a.astype(float)
    h, w = rng.integers(20, 80, 2); y, x = rng.integers(0, S - h), rng.integers(0, S - w)
    out[y:y + h, x:x + w] += rng.normal(0, rng.uniform(30, 60), (h, w))
    return np.clip(out, 0, 255).astype(np.uint8)


def glitch_saturation(a, b, rng):
    """Over-exposed (clipped white) blob."""
    yy, xx = np.mgrid[0:S, 0:S]
    cy, cx = rng.uniform(30, S - 30, 2); r = rng.uniform(8, 25)
    out = a.copy(); out[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 255
    return out


def domain_pivots(a, b, rng):
    """Centre-pivot irrigation circles on a flat background (a classic Earth-from-orbit pattern)."""
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    out = np.full((S, S), rng.uniform(100, 160))
    for _ in range(rng.integers(3, 8)):
        cy, cx = rng.uniform(0, S, 2); r = rng.uniform(20, 55)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        out[m] = rng.uniform(40, 220)
        if rng.random() < .5:   # a harvested wedge
            ang = np.arctan2(yy - cy, xx - cx); a0 = rng.uniform(-np.pi, np.pi)
            out[m & (np.abs(np.angle(np.exp(1j * (ang - a0)))) < rng.uniform(.3, 1.2))] = rng.uniform(40, 220)
    out += rng.normal(0, 6, (S, S))
    return np.clip(ndimage.gaussian_filter(out, .8), 0, 255).astype(np.uint8)


def domain_fields(a, b, rng):
    """Agricultural patchwork: Voronoi polygons of flat tone with fine row texture."""
    yy, xx = np.mgrid[0:S, 0:S].astype(float)
    k = rng.integers(8, 20); c = rng.uniform(0, S, (k, 2))
    lab = np.argmin((yy[..., None] - c[:, 0]) ** 2 + (xx[..., None] - c[:, 1]) ** 2, -1)
    tone, ang = rng.uniform(50, 210, k), rng.uniform(0, np.pi, k)
    rows = np.cos(2 * np.pi * (xx * np.cos(ang[lab]) + yy * np.sin(ang[lab])) / rng.uniform(3, 6))
    out = tone[lab] + rng.uniform(4, 12) * rows + rng.normal(0, 5, (S, S))
    return np.clip(out, 0, 255).astype(np.uint8)


NEW_GENERATORS = {"splice_matched": ("splice", splice_matched), "splice_feather": ("splice", splice_feather),
                  "glitch_noise": ("glitch", glitch_noise), "glitch_saturation": ("glitch", glitch_saturation),
                  "domain_pivots": ("domain", domain_pivots), "domain_fields": ("domain", domain_fields)}


def make_set(X, nodata_frac, gens, n, seed):
    rng = np.random.default_rng(seed)
    clean = np.where(nodata_frac < 0.001)[0]
    src, partner = rng.choice(clean, n, replace=False), rng.choice(clean, n, replace=False)
    imgs, rows = [], []
    for name, (cls, fn) in gens.items():
        for i, j in zip(src, partner):
            imgs.append(fn(X[i], X[j], rng)); rows.append(dict(type=name, cls=cls, source=int(i)))
    return np.stack(imgs), pd.DataFrame(rows)


def load_data():
    """Crops and no-data masks as cached by the main notebook; rebuilt from data/ if the cache is missing."""
    fx, fm = os.path.join(CACHE, "images_u8.npy"), os.path.join(CACHE, "nodata_masks.npy")
    if not os.path.exists(fx):
        os.makedirs(CACHE, exist_ok=True)
        load_images(list_images(os.path.join(ROOT, "data", "images_unz", "images")), cache_path=fx)
    X = np.load(fx)
    return X, compute_nodata_masks(X, fm)


def load_probes(X):
    """dev = submitted benchmark (seed 0); test = seed-1 same types + seed-2 new types. Cached with masks."""
    f = os.path.join(R2, "cache", "probes.npz")
    if os.path.exists(f):
        d = np.load(f, allow_pickle=True)
        return {k: (d[k + "_X"], d[k + "_M"], pd.DataFrame(json.loads(str(d[k + "_meta"])))) for k in ("dev", "test")}
    nd = (X <= 2).mean((1, 2))
    Pd, md = B.make_probes(X, nd, 300, seed=0)
    Pt1, mt1 = make_set(X, nd, B.GENERATORS, 300, seed=1)
    Pt2, mt2 = make_set(X, nd, NEW_GENERATORS, 300, seed=2)
    Pt = np.concatenate([Pt1, Pt2]); mt = pd.concat([mt1, mt2], ignore_index=True)
    mt["held_out_type"] = mt.type.isin(NEW_GENERATORS)
    md["held_out_type"] = False
    out = {}
    for k, P, m in (("dev", Pd, md), ("test", Pt, mt)):
        out[k] = (P, compute_nodata_masks(P), m)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savez(f, **{f"{k}_X": v[0] for k, v in out.items()}, **{f"{k}_M": v[1] for k, v in out.items()},
             **{f"{k}_meta": json.dumps(v[2].to_dict("records")) for k, v in out.items()})
    return out


# ------------------------------------------------------------ stage-2 inputs
@torch.no_grad()
def stage2_images(model, images_u8, masks_u8=None, kind="signed", gain=4.0, batch_size=128):
    """signed : 128 + 127 clip(gain r)                  (v6, as submitted)
       abs    : 255 clip(gain |r|)
       energy : log10 of the Gaussian-smoothed squared residual (sigma 2), [1e-5, 1] -> [0, 255],
                i.e. the Phase-3 error heatmap itself, as an image"""
    model.eval().to(DEVICE)
    X = torch.from_numpy(images_u8).to(DEVICE)
    Mt = None if masks_u8 is None else torch.from_numpy(masks_u8).to(DEVICE)
    win = _gauss_window(13, 2.0, DEVICE)
    out = np.empty_like(images_u8)
    for s in range(0, len(X), batch_size):
        idx = torch.arange(s, min(s + batch_size, len(X)), device=DEVICE)
        x, w = to_batch(X, idx, False, Mt)
        r = x - model(x)[0].float()
        if w is not None:
            r = r * w
        if kind == "signed":
            u = (128 + 127 * (gain * r).clamp(-1, 1)) / 255
        elif kind == "abs":
            u = (gain * r.abs()).clamp(0, 1)
        elif kind == "energy":
            e = F.conv2d(r * r, win, padding=6)
            if w is not None:   # validity-weighted smoothing; no-data -> floor
                e = e / F.conv2d(w, win, padding=6).clamp_min(1e-3) * w
            u = ((torch.log10(e + 1e-5) + 5) / 5).clamp(0, 1)
        elif kind == "dssim":   # structural dissimilarity (1 - SSIM)/2 in [0, 1], x2 so typical failures span the range
            xh = model(x)[0].float()
            d = (1 - ssim_map(xh, x)) / 2
            if w is not None:
                d = d * w
            u = (2 * d).clamp(0, 1)
        elif kind == "grad":    # magnitude of the edge (finite-difference gradient) residual
            gx = F.pad(r[..., :, 1:] - r[..., :, :-1], (0, 1, 0, 0))
            gy = F.pad(r[..., 1:, :] - r[..., :-1, :], (0, 0, 0, 1))
            u = (gain * torch.sqrt(gx * gx + gy * gy)).clamp(0, 1)
        elif kind == "absdil":  # |r| max-filtered over 5x5 so small defects grow
            u = (gain * F.max_pool2d(r.abs(), 5, stride=1, padding=2)).clamp(0, 1)
        else:
            raise ValueError(kind)
        out[s:s + len(idx)] = (255 * u).round().squeeze(1).byte().cpu().numpy()
    return out


# ------------------------------------------------------------ evaluation
def eval_scores(nov_real, nov_probe, meta, thr=None):
    thr = mad_threshold(nov_real, 3.5) if thr is None else thr
    rows = []
    for t, g in meta.groupby("type", sort=False):
        p = nov_probe[g.index.values]
        y = np.r_[np.zeros(len(nov_real)), np.ones(len(p))]
        rows.append(dict(type=t, cls=g.cls.iloc[0], auroc=roc_auc_score(y, np.r_[nov_real, p]),
                         recall=float((p > thr).mean()),
                         held_out=bool(g["held_out_type"].iloc[0]) if "held_out_type" in g else False))
    return pd.DataFrame(rows)


SPOT = {9828: "Pentagon", 6476: "orchard", 7971: "grid", 7701: "runway", 5508: "buildings",
        15: "black bar", 503: "two streaks", 676: "dead blocks"}


def spot_ranks(nov_real):
    rank = (-nov_real).argsort().argsort() + 1
    return {lab: int(rank[k - 1]) for k, lab in SPOT.items()}


def summary_line(det):
    c = det.groupby("cls").auroc.mean()
    return dict(mean=det.auroc.mean(), splice=c.get("splice"), glitch=c.get("glitch"),
                domain=c.get("domain"), recall=det.recall.mean())
