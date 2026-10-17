"""Round-3 search harness on the faithful injection benchmark. Saves every latent so that latent
compositions and Isolation-Forest settings are compared offline (p3_eval.py).

Stage-1 / stage-2 training modes
  std     : nssc_core.train_autoencoder (identical to p2.py)
  trim    : per-sample loss trimming after a warm-up (drop the top q of per-sample losses per batch)
  refine  : train std, score every crop, drop crops that the SAME modified-z rule (k = 3.5) flags,
            retrain from scratch on the rest, embed all crops. No contamination fraction is assumed.
            --refine-score recon : per-crop stage-1 loss
            --refine-score v7    : Isolation-Forest novelty on [signed | abs] codes of the first pass

usage: python post_submission_improvements/iteration_v7/code/p3.py --tag base_s100 --inject dev --seed 100 --s2 signed,abs,dssim,grad,absdil
"""
import argparse, os, json, time, math
import numpy as np, pandas as pd, torch
from r2lib import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--inject", default="dev", choices=["dev", "new"])
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--per-type", type=int, default=30)
ap.add_argument("--s1-epochs", type=int, default=30)
ap.add_argument("--s1-latent", type=int, default=256)
ap.add_argument("--s2", default="signed,abs")
ap.add_argument("--s2-latent", type=int, default=64)
ap.add_argument("--s2-epochs", type=int, default=30)
ap.add_argument("--mode", default="std", choices=["std", "trim", "refine"])
ap.add_argument("--trim-q", type=float, default=0.05)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--refine-score", default="recon", choices=["recon", "v7"])
ap.add_argument("--refine-k", type=float, default=3.5)
args = ap.parse_args()
OUT = os.path.join(R2, "p3", args.tag); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
S1W = dict(l1=0.16, ssim=0.84, grad=0.1)


# ------------------------------------------------------------ per-sample losses and a trimmed trainer
def per_sample_loss(xh, x, w, kind):
    """Per-sample version of ReconLoss on valid pixels: kind 's1' = 0.84(1-SSIM)+0.16 L1+0.1 grad-L1, 's2' = MSE+0.5 L1."""
    if w is None:
        w = torch.ones_like(x)
    den = w.sum((1, 2, 3)).clamp_min(1.0)
    l1 = ((xh - x).abs() * w).sum((1, 2, 3)) / den
    if kind == "s2":
        return (((xh - x) ** 2) * w).sum((1, 2, 3)) / den + 0.5 * l1
    ss = 1 - (ssim_map(xh, x) * w).sum((1, 2, 3)) / den
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    wx, wy = w[..., :, 1:] * w[..., :, :-1], w[..., 1:, :] * w[..., :-1, :]
    g = ((dx(xh) - dx(x)).abs() * wx).sum((1, 2, 3)) / wx.sum((1, 2, 3)).clamp_min(1.0) + \
        ((dy(xh) - dy(x)).abs() * wy).sum((1, 2, 3)) / wy.sum((1, 2, 3)).clamp_min(1.0)
    return S1W["ssim"] * ss + S1W["l1"] * l1 + S1W["grad"] * g


def train_trimmed(model, images_u8, kind, epochs, masks_u8=None, q=0.05, warmup=5, lr=1e-3, batch_size=64, seed=42):
    """train_autoencoder with the same optimiser/schedule/augmentation, but after `warmup` epochs the
    highest-loss fraction q of each batch is excluded from the gradient (small-loss trick)."""
    seed_everything(seed)
    model = model.to(DEVICE)
    X = torch.from_numpy(images_u8).to(DEVICE)
    Mt = None if masks_u8 is None else torch.from_numpy(masks_u8).to(DEVICE)
    N = len(X)
    perm = np.random.permutation(N)
    tr = perm[int(N * 0.1):]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = math.ceil(len(tr) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.1)
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(tr)
        run = 0.0
        for s in range(steps):
            idx = torch.from_numpy(order[s * batch_size:(s + 1) * batch_size]).to(DEVICE)
            x, w = to_batch(X, idx, True, Mt)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                xh = model(x)[0]
            li = per_sample_loss(xh.float(), x, w, kind)
            if ep >= warmup and q > 0:
                keep = max(1, int(round(len(li) * (1 - q))))
                li = torch.sort(li)[0][:keep]
            loss = li.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            run += loss.item()
        if (ep + 1) % 10 == 0:
            log(f"  trimmed {kind} ep {ep + 1}/{epochs} loss {run / steps:.4f}")
    return model.eval()


@torch.no_grad()
def crop_losses(model, images_u8, masks_u8, kind="s1", bs=128):
    model.eval()
    X = torch.from_numpy(images_u8).to(DEVICE)
    Mt = None if masks_u8 is None else torch.from_numpy(masks_u8).to(DEVICE)
    out = []
    for s in range(0, len(X), bs):
        idx = torch.arange(s, min(s + bs, len(X)), device=DEVICE)
        x, w = to_batch(X, idx, False, Mt)
        out.append(per_sample_loss(model(x)[0].float(), x, w, kind).cpu().numpy())
    return np.concatenate(out)


def fit(kind, latent, images_u8, masks_u8, epochs, train_idx=None):
    """Train one autoencoder in the run's mode, on train_idx (default: all crops)."""
    I = np.arange(len(images_u8)) if train_idx is None else train_idx
    seed_everything(42)
    model = ConvAE(latent_dim=latent)
    if args.mode == "trim":
        return train_trimmed(model, images_u8[I], kind, epochs, masks_u8[I], q=args.trim_q, warmup=args.warmup)
    loss = ReconLoss(**CONFIGS["v3"]["loss"]) if kind == "s1" else ReconLoss(w_mse=1.0, w_l1=0.5)
    model, _, _ = train_autoencoder(model, images_u8[I], loss, epochs=epochs, lr=1e-3, masks_u8=masks_u8[I], log_every=1000)
    return model.eval()


# ------------------------------------------------------------ contaminate (identical to p2.py for the same seed)
X, M = load_data()
nd = (X <= 2).mean((1, 2))
gens = B.GENERATORS if args.inject == "dev" else NEW_GENERATORS
rng = np.random.default_rng(args.seed)
clean = np.setdiff1d(np.where(nd < 0.001)[0], [k - 1 for k in SPOT])
n = args.per_type * len(gens)
pick = rng.choice(clean, 2 * n, replace=False)
src, partner = pick[:n], pick[n:]
Xc, lab = X.copy(), np.array([""] * len(X), dtype=object)
for g, (name, (cls, fn)) in enumerate(gens.items()):
    for i, j in zip(src[g * args.per_type:(g + 1) * args.per_type], partner[g * args.per_type:(g + 1) * args.per_type]):
        Xc[i] = fn(X[i], X[j], rng); lab[i] = name
Mc = M.copy(); Mc[src] = compute_nodata_masks(Xc[src])
Xin = fill_nodata(Xc, Mc)
log(f"injected {n} fakes ({n / len(X):.1%}), mode {args.mode}")
kinds = [k for k in args.s2.split(",") if k]


def pipeline(train_idx=None, tagp=""):
    s1 = fit("s1", args.s1_latent, Xin, Mc, args.s1_epochs, train_idx)
    Z = {"Z1": embed_and_reconstruct(s1, Xin, Mc)[0], "s1loss": crop_losses(s1, Xin, Mc)}
    log(f"{tagp}stage 1 done")
    for kd in kinds:
        R = stage2_images(s1, Xin, Mc, kd)
        s2 = fit("s2", args.s2_latent, R, Mc, args.s2_epochs, train_idx)
        Z[f"Z2_{kd}"] = embed_and_reconstruct(s2, R, Mc)[0]
        log(f"{tagp}stage 2 {kd} done")
    return Z


Z = pipeline()
meta = dict(vars(args), n_injected=int(n), refined_dropped=0)
if args.mode == "refine":
    if args.refine_score == "recon":
        sc = Z["s1loss"]
    else:
        assert "signed" in kinds and "abs" in kinds
        sc = novelty_scores(np.hstack([Z["Z2_signed"], Z["Z2_abs"]]))[0]
    drop = sc > mad_threshold(sc, args.refine_k)
    keep_idx = np.where(~drop)[0]
    meta.update(refined_dropped=int(drop.sum()), refined_dropped_injected=int((drop & (lab != "")).sum()))
    log(f"refine ({args.refine_score}): dropping {drop.sum()} crops from training, {(drop & (lab != '')).sum()} of them injected")
    for k, v in Z.items():
        np.save(os.path.join(OUT, f"pass1_{k}.npy"), v) if k.startswith("Z2_signed") or k.startswith("Z2_abs") else None
    Z = pipeline(keep_idx, "refined ")
np.savez(os.path.join(OUT, "latents.npz"), lab=lab.astype(str), **Z)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
log(f"saved {OUT}")
