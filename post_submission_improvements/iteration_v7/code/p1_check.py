"""Needs the 100-epoch runs B2 and E1 first (run.py --tag B2 / --tag E1); their weights are not included.

Unseen-vs-unseen check: probes built only from the 10% of crops no autoencoder trained on,
compared only against those same held-out crops. Removes the seen-vs-unseen (memorisation) advantage."""
import numpy as np, os
from r2lib import *

seed_everything(42); perm = np.random.permutation(10422); val = np.zeros(10422, bool); val[perm[:1042]] = True
X, M = load_data(); Xin = fill_nodata(X, M)
nd = (X <= 2).mean((1, 2)); nd_val = np.where(val, nd, 1.0)          # only held-out crops are eligible
print("clean held-out crops:", int((nd_val < 0.001).sum()))
P, meta = make_set(X, nd_val, B.GENERATORS, 300, seed=10)
PM = compute_nodata_masks(P); Pin = fill_nodata(P, PM)

def load(path, **kw):
    m = ConvAE(**kw); m.load_state_dict(torch.load(path, map_location=DEVICE)); return m.to(DEVICE).eval()

out = os.path.join(ROOT, "outputs")
models = {"v6 submitted": (load_model("v3", out), [("signed", load_model("v6", out))], True),
          "E1 (100-ep s1, signed+abs)": (load(os.path.join(R2, "runs/B2/s1.pt"), latent_dim=256),
                                         [("signed", load(os.path.join(R2, "runs/E1/s2_signed64e30.pt"), latent_dim=64)),
                                          ("abs", load(os.path.join(R2, "runs/E1/s2_abs64e30.pt"), latent_dim=64))], False)}
for name, (s1, s2s, with_scene) in models.items():
    feats = {}
    for k, (A, AM) in {"real": (Xin, M), "probe": (Pin, PM)}.items():
        parts = [embed_and_reconstruct(s1, A, AM)[0]] if with_scene else []
        parts += [embed_and_reconstruct(m, stage2_images(s1, A, AM, kind), AM)[0] for kind, m in s2s]
        feats[k] = np.hstack(parts)
    sc = B.forest_scorer(feats["real"]); nr, npb = sc(feats["real"]), sc(feats["probe"])
    det = eval_scores(nr[val], npb, meta)            # negatives = held-out real crops only
    s = summary_line(det)
    print(f"{name:28s} unseen-vs-unseen mean {s['mean']:.3f} | splice {s['splice']:.3f} glitch {s['glitch']:.3f} domain {s['domain']:.3f}")
