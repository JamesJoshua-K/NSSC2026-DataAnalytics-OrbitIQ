"""Offline scoring of saved p3 latents: any latent composition x any Isolation-Forest setting.

combos  : '+'-joined blocks; a block is Z1 | <stage-2 kind> | pcaK:Z1   e.g.  signed+abs,  pca16:Z1+signed+abs
forests : n_estimators x max_samples x max_features, e.g.  500x256x1.0
usage   : python post_submission_improvements/iteration_v7/code/p3_eval.py base_s100 base_s101 base_s102 --combos "signed+abs,abs+dssim" --forests 500x256x1.0 --out r3_base.csv
"""
import argparse, os, json, glob
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from r2lib import R2, B, NEW_GENERATORS, mad_threshold, spot_ranks

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--combos", default="signed+abs")
ap.add_argument("--forests", default="500x256x1.0")
ap.add_argument("--out", default="")
args = ap.parse_args()


def forest(Z, n_est, psi, maxf, seeds=(0, 1, 2, 3, 4)):
    return np.mean([-IsolationForest(n_estimators=n_est, max_samples=psi, max_features=maxf, contamination="auto",
                                     random_state=s, n_jobs=-1).fit(Z).score_samples(Z) for s in seeds], 0)


def block(L, b):
    if b.startswith("pca"):
        k, src = b[3:].split(":")
        return PCA(int(k), random_state=0).fit_transform(L[src])
    return L[b] if b == "Z1" else L[f"Z2_{b}"]


rows = []
for run in args.runs:
    d = os.path.join(R2, "p3", run)
    L = np.load(os.path.join(d, "latents.npz"), allow_pickle=True)
    meta = json.load(open(os.path.join(d, "meta.json")))
    gens = B.GENERATORS if meta["inject"] == "dev" else NEW_GENERATORS
    lab = L["lab"]
    neg = lab == ""
    for combo in args.combos.split(","):
        try:
            Zr = np.hstack([block(L, b) for b in combo.split("+")])
        except KeyError as e:
            print(f"{run}: missing block {e} for {combo}"); continue
        for fs in args.forests.split(","):
            n_est, psi, maxf = fs.split("x")
            nov = forest(Zr, int(n_est), int(psi), float(maxf))
            thr = mad_threshold(nov)
            fl = nov > thr
            r = dict(run=run, inject=meta["inject"], seed=meta["seed"], mode=meta["mode"], combo=combo, forest=fs, dim=Zr.shape[1],
                     n_flag=int(fl.sum()), recall=float(fl[~neg].mean()), precision=float((~neg[fl]).mean()) if fl.any() else 0.0,
                     **spot_ranks(nov))
            for name, (cls, _) in gens.items():
                pos = lab == name
                r[f"t_{name}"] = roc_auc_score(np.r_[np.zeros(neg.sum()), np.ones(pos.sum())], np.r_[nov[neg], nov[pos]])
            for c in ("splice", "glitch", "domain"):
                r[c] = np.mean([r[f"t_{k}"] for k, (cl, _) in gens.items() if cl == c])
            r["mean"] = np.mean([r[f"t_{k}"] for k in gens])
            rows.append(r)
            print(f"{run:16s} {combo:24s} {fs:12s} d={Zr.shape[1]:3d} mean {r['mean']:.3f} | sp {r['splice']:.3f} gl {r['glitch']:.3f} "
                  f"dom {r['domain']:.3f} | flags {r['n_flag']:4d} rec {r['recall']:.2f} prec {r['precision']:.2f} | "
                  f"Earth {r['orchard']},{r['grid']},{r['runway']},{r['Pentagon']} bar {r['black bar']}", flush=True)

df = pd.DataFrame(rows)
if args.out:
    f = os.path.join(R2, "results", args.out)
    if os.path.exists(f):
        old = pd.read_csv(f)
        key = ["run", "combo", "forest"]
        df = pd.concat([old[~old.set_index(key).index.isin(df.set_index(key).index)], df])
    df.to_csv(f, index=False)
