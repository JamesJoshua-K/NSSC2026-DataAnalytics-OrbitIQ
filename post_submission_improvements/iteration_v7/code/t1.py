"""Final test T1 (run once): v6 exactly as submitted vs v7 = submitted v3 + submitted v6 stage 2
(signed residual) + the new 64-d absolute-residual autoencoder (stage 2b).
The reported T1 numbers used an earlier training of stage 2b with the same recipe (run D2, not included);
with weights/stage2b_abs64.pt, the network behind the real-data results, they can differ slightly.
Also: the seen-vs-unseen check for both (should be ~0.5)."""
import argparse, os, json
import numpy as np
from r2lib import *

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="V7"); args = ap.parse_args()
X, M = load_data(); Xin = fill_nodata(X, M)
P = load_probes(X)
out = os.path.join(ROOT, "outputs")
s1, s2a = load_model("v3", out), load_model("v6", out)
s2b = ConvAE(latent_dim=64); s2b.load_state_dict(torch.load(os.path.join(R2, "weights", "stage2b_abs64.pt"), map_location=DEVICE))
s2b = s2b.to(DEVICE).eval()
lat = {}
for k, (A, AM) in {"real": (Xin, M), **{k: (fill_nodata(v[0], v[1]), v[1]) for k, v in P.items()}}.items():
    Z, mse, _ = embed_and_reconstruct(s1, A, AM)
    lat[f"Z1_{k}"], lat[f"mse_{k}"] = Z, mse
    lat[f"Z2_signed64e30_{k}"] = embed_and_reconstruct(s2a, residual_images(s1, A, AM), AM)[0]
    lat[f"Z2_abs64e30_{k}"] = embed_and_reconstruct(s2b, stage2_images(s1, A, AM, "abs"), AM)[0]
OUT = os.path.join(R2, "runs", args.tag); os.makedirs(OUT, exist_ok=True)
np.savez(os.path.join(OUT, "latents.npz"), **lat)
sub = np.load(os.path.join(out, "v6/latents.npy"))
print("v6 part identical to submission:", float(np.abs(np.hstack([lat["Z1_real"], lat["Z2_signed64e30_real"]]) - sub).max()))

seed_everything(42); perm = np.random.permutation(10422); val = np.zeros(10422, bool); val[perm[:1042]] = True
for name, parts in {"v6 submitted": ("Z1", "Z2_signed64e30"), "v7 [signed|abs]": ("Z2_signed64e30", "Z2_abs64e30")}.items():
    Zr = np.hstack([lat[f"{p}_real"] for p in parts])
    nov = B.forest_scorer(Zr)(Zr)
    print(f"{name:18s} seen-vs-unseen AUROC {roc_auc_score(val, nov):.3f}  (0.5 = no memorisation advantage)")
