"""Faithful benchmark: fakes are injected INTO the dataset (replacing the crop they were made from,
about 2.3% contamination), the whole pipeline is retrained on the contaminated data, and the
injected crops are scored against the rest - exactly the situation of the organisers' data.

usage: python post_submission_improvements/iteration_v7/code/p2.py --tag P30_d0 --s1-epochs 30 --inject dev --seed 100
"""
import argparse, os, json, time
import numpy as np, pandas as pd, torch
from sklearn.decomposition import PCA
from r2lib import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--s1-epochs", type=int, default=30)
ap.add_argument("--inject", default="dev", choices=["dev", "new"])
ap.add_argument("--per-type", type=int, default=30)
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--s2-epochs", type=int, default=30)
args = ap.parse_args()
OUT = os.path.join(R2, "p2", args.tag); os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# ---------------- contaminate
X, M = load_data()
nd = (X <= 2).mean((1, 2))
gens = B.GENERATORS if args.inject == "dev" else NEW_GENERATORS
rng = np.random.default_rng(args.seed)
clean = np.setdiff1d(np.where(nd < 0.001)[0], [k - 1 for k in SPOT])     # never overwrite the real spot-check crops
n = args.per_type * len(gens)
pick = rng.choice(clean, 2 * n, replace=False)
src, partner = pick[:n], pick[n:]
Xc, lab = X.copy(), np.array([""] * len(X), dtype=object)
for g, (name, (cls, fn)) in enumerate(gens.items()):
    for i, j in zip(src[g * args.per_type:(g + 1) * args.per_type], partner[g * args.per_type:(g + 1) * args.per_type]):
        Xc[i] = fn(X[i], X[j], rng); lab[i] = name
Mc = M.copy(); Mc[src] = compute_nodata_masks(Xc[src])
Xin = fill_nodata(Xc, Mc)
print(f"injected {n} fakes ({n / len(X):.1%}) | {time.time()-t0:.0f}s", flush=True)

# ---------------- retrain the pipeline on the contaminated data
seed_everything(42)
s1, h1, _ = train_autoencoder(ConvAE(latent_dim=256), Xin, ReconLoss(**CONFIGS["v3"]["loss"]),
                              epochs=args.s1_epochs, lr=1e-3, masks_u8=Mc, log_every=10)
s1.eval()
Z = {"Z1": embed_and_reconstruct(s1, Xin, Mc)[0]}
for kind in ("signed", "abs"):
    R = stage2_images(s1, Xin, Mc, kind)
    seed_everything(42)
    s2, _, _ = train_autoencoder(ConvAE(latent_dim=64), R, ReconLoss(w_mse=1.0, w_l1=0.5), epochs=args.s2_epochs, lr=1e-3,
                                 masks_u8=Mc, log_every=10)
    Z[kind] = embed_and_reconstruct(s2.eval(), R, Mc)[0]
Z["Z1eq"] = PCA(64, random_state=0).fit_transform(Z["Z1"])
print(f"pipeline retrained | {time.time()-t0:.0f}s", flush=True)

# ---------------- score
COMBOS = {"v6 recipe [Z1|signed]": ("Z1", "signed"), "[signed|abs]": ("signed", "abs"),
          "[eqZ1|signed|abs]": ("Z1eq", "signed", "abs"), "[Z1|signed|abs]": ("Z1", "signed", "abs"),
          "[signed]": ("signed",), "[abs]": ("abs",), "[Z1]": ("Z1",)}
neg = lab == ""
rows = []
for cname, parts in COMBOS.items():
    nov = novelty_scores(np.hstack([Z[p] for p in parts]))[0]
    thr = mad_threshold(nov)
    r = dict(tag=args.tag, s1_epochs=args.s1_epochs, inject=args.inject, seed=args.seed, combo=cname,
             n_flag=int((nov > thr).sum()), recall=float((nov[~neg] > thr).mean()),
             precision=float((~neg[nov > thr]).mean()) if (nov > thr).any() else 0.0, **spot_ranks(nov))
    for name, (cls, _) in gens.items():
        pos = lab == name
        r[f"t_{name}"] = roc_auc_score(np.r_[np.zeros(neg.sum()), np.ones(pos.sum())], np.r_[nov[neg], nov[pos]])
    for c in ("splice", "glitch", "domain"):
        r[c] = np.mean([r[f"t_{k}"] for k, (cl, _) in gens.items() if cl == c])
    r["mean"] = np.mean([r[f"t_{k}"] for k in gens])
    rows.append(r)
    print(f"{args.tag:8s} {cname:22s} mean {r['mean']:.3f} | sp {r['splice']:.3f} gl {r['glitch']:.3f} "
          f"dom {r['domain']:.3f} | flags {r['n_flag']:4d} recall {r['recall']:.2f} precision {r['precision']:.2f} | "
          f"Earth {r['orchard']},{r['grid']},{r['runway']},{r['Pentagon']} bar {r['black bar']}", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "results.csv"), index=False)
print(f"done | {time.time()-t0:.0f}s", flush=True)
