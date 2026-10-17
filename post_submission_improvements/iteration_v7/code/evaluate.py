"""Score every latent combination of the given runs with the unchanged Isolation Forest + threshold.

usage: python post_submission_improvements/iteration_v7/code/evaluate.py A1 A2 ...            (dev benchmark: used for selection)
       python post_submission_improvements/iteration_v7/code/evaluate.py A4 --split test      (held-out benchmark: final report only)
"""
import argparse, os, glob
import numpy as np, pandas as pd
from r2lib import *
from sklearn.decomposition import PCA

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="*")
ap.add_argument("--split", default="dev")
ap.add_argument("--combos", default="", help="comma list to restrict, e.g. 'Z2:energy64e30,Z1+Z2:energy64e30'")
args = ap.parse_args()

X = np.load(os.path.join(CACHE, "images_u8.npy"), mmap_mode="r")
meta = load_probes(X)[args.split][2]
runs = args.runs or sorted(os.path.basename(p) for p in glob.glob(os.path.join(R2, "runs", "*")))
rows = []
for tag in runs:
    f = os.path.join(R2, "runs", tag, "latents.npz")
    if not os.path.exists(f):
        print(f"{tag}: no latents yet"); continue
    L = np.load(f)
    s2 = sorted({k[3:-5] for k in L.files if k.startswith("Z2_") and k.endswith("_real")})
    combos = {"Z1": ("Z1",)}
    for n in s2:
        combos[f"Z2:{n}"] = (f"Z2_{n}",)
        combos[f"Z1+Z2:{n}"] = ("Z1", f"Z2_{n}")
        if L["Z1_real"].shape[1] > L[f"Z2_{n}_real"].shape[1]:
            combos[f"eq(Z1)+Z2:{n}"] = ("Z1eq", f"Z2_{n}")   # scene code PCA-reduced to the residual code's size
    for i, n1 in enumerate(s2):                      # pairs of residual codes (e.g. signed + magnitude)
        for n2 in s2[i + 1:]:
            combos[f"Z2:{n1}+{n2}"] = (f"Z2_{n1}", f"Z2_{n2}")
            combos[f"Z1+Z2:{n1}+{n2}"] = ("Z1", f"Z2_{n1}", f"Z2_{n2}")
            if L["Z1_real"].shape[1] > L[f"Z2_{n1}_real"].shape[1]:
                combos[f"eq(Z1)+Z2:{n1}+{n2}"] = ("Z1eq", f"Z2_{n1}", f"Z2_{n2}")
    if args.combos:
        combos = {k: v for k, v in combos.items() if k in args.combos.split(",")}
    for cname, parts in combos.items():
        def get(p, split):
            if p != "Z1eq":
                return L[f"{p}_{split}"]
            d = L[f"{parts[1]}_real"].shape[1]
            pca = PCA(d, random_state=0).fit(L["Z1_real"])    # fitted on real crops only
            return pca.transform(L[f"Z1_{split}"])
        Zr = np.hstack([get(p, "real") for p in parts])
        Zp = np.hstack([get(p, args.split) for p in parts])
        score = B.forest_scorer(Zr)
        nr, npb = score(Zr), score(Zp)
        det = eval_scores(nr, npb, meta)
        r = dict(run=tag, combo=cname, dim=Zr.shape[1], **summary_line(det),
                 n_flag=int((nr > mad_threshold(nr)).sum()), **spot_ranks(nr))
        if args.split == "test":
            r["mean_same_types"] = det[~det.held_out].auroc.mean()
            r["mean_new_types"] = det[det.held_out].auroc.mean()
        r.update({f"t_{t}": a for t, a in zip(det.type, det.auroc)})
        rows.append(r)
        print(f"{tag:6s} {cname:24s} d={Zr.shape[1]:3d} mean {r['mean']:.3f} | sp {r['splice']:.3f} "
              f"gl {r['glitch']:.3f} dom {r['domain']:.3f} | flags {r['n_flag']:4d} | Earth "
              f"{r['orchard']},{r['grid']},{r['runway']},{r['Pentagon']} bar {r['black bar']} "
              f"streaks {r['two streaks']}", flush=True)
    # reconstruction-error reference (not a legal latent, context only)
    det = eval_scores(L["mse_real"], L[f"mse_{args.split}"], meta)
    print(f"{tag:6s} {'(recon MSE reference)':24s}       mean {det.auroc.mean():.3f}", flush=True)

if not rows:
    raise SystemExit
out = pd.DataFrame(rows)
f = os.path.join(R2, "results", f"results_{args.split}.csv")
if os.path.exists(f) and os.path.getsize(f) > 0:
    old = pd.read_csv(f)
    out = pd.concat([old[~old.set_index(["run", "combo"]).index.isin(out.set_index(["run", "combo"]).index)], out])
out.to_csv(f, index=False)
