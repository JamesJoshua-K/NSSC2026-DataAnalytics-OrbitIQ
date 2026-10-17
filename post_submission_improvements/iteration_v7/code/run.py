"""Train one stage-1 variant (or reuse the submitted v3) and one or more stage-2 variants on its
residual, then cache every latent for real crops + dev probes + test probes.

usage: python post_submission_improvements/iteration_v7/code/run.py --tag A1 --s1 v3 --s2 signed:64:30
       python post_submission_improvements/iteration_v7/code/run.py --tag B2 --s1 new --s1-latent 256 --s1-epochs 100 --s2 signed:64:30,energy:64:30
"""
import argparse, os, json, time
import numpy as np, pandas as pd, torch
from r2lib import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--s1", default="v3", help="'v3' = submitted checkpoint, 'new' = train")
ap.add_argument("--s1-latent", type=int, default=256)
ap.add_argument("--s1-epochs", type=int, default=30)
ap.add_argument("--s1-chans", default="32,64,128,256,256")
ap.add_argument("--s1-res", type=int, default=0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--s2", default="signed:64:30", help="comma list of kind:latent:epochs[:gain]")
args = ap.parse_args()

OUT = os.path.join(R2, "runs", args.tag)
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
X, M = load_data()
Xin = fill_nodata(X, M)
P = load_probes(X)
Pin = {k: (fill_nodata(v[0], v[1]), v[1]) for k, v in P.items()}
print(f"data {X.shape}, dev {len(P['dev'][0])}, test {len(P['test'][0])} probes | {time.time()-t0:.0f}s", flush=True)

# ---------------- stage 1
S1LOSS = CONFIGS["v3"]["loss"]
if args.s1 == "v3":
    s1 = load_model("v3", os.path.join(ROOT, "outputs"))
elif args.s1.startswith("run:"):        # reuse the stage 1 trained by another run
    src = os.path.join(R2, "runs", args.s1[4:])
    a = json.load(open(os.path.join(src, "args.json")))
    s1 = ConvAE(latent_dim=a["s1_latent"], chans=tuple(int(c) for c in a["s1_chans"].split(",")), res_blocks=a["s1_res"])
    s1.load_state_dict(torch.load(os.path.join(src, "s1.pt"), map_location=DEVICE)); s1 = s1.to(DEVICE)
else:
    chans = tuple(int(c) for c in args.s1_chans.split(","))
    f = os.path.join(OUT, "s1.pt")
    seed_everything(args.seed)
    s1 = ConvAE(latent_dim=args.s1_latent, chans=chans, res_blocks=args.s1_res)
    if os.path.exists(f):
        s1.load_state_dict(torch.load(f, map_location=DEVICE)); s1 = s1.to(DEVICE)
    else:
        s1, hist, _ = train_autoencoder(s1, Xin, ReconLoss(**S1LOSS), epochs=args.s1_epochs, lr=1e-3,
                                        masks_u8=M, log_every=5, seed=args.seed)
        torch.save(s1.state_dict(), f); hist.to_csv(os.path.join(OUT, "s1_history.csv"), index=False)
s1.eval()
lat = {}
for k, (A, AM) in {"real": (Xin, M), **Pin}.items():
    Z, mse, ssim = embed_and_reconstruct(s1, A, AM)
    lat[f"Z1_{k}"], lat[f"mse_{k}"], lat[f"ssim_{k}"] = Z, mse, ssim
print(f"stage 1 done | val-free recon MSE (real) {lat['mse_real'].mean():.5f} | {time.time()-t0:.0f}s", flush=True)

# ---------------- stage 2 variants
for spec in args.s2.split(","):
    if not spec or spec == "none":
        continue
    parts = spec.split(":")
    kind, s2lat, s2ep = parts[0], int(parts[1]), int(parts[2])
    gain = float(parts[3]) if len(parts) > 3 else 4.0
    name = f"{kind}{s2lat}e{s2ep}" + (f"g{gain:g}" if len(parts) > 3 else "")
    R_real = stage2_images(s1, Xin, M, kind, gain)
    f = os.path.join(OUT, f"s2_{name}.pt")
    seed_everything(args.seed)
    s2 = ConvAE(latent_dim=s2lat)
    if os.path.exists(f):
        s2.load_state_dict(torch.load(f, map_location=DEVICE)); s2 = s2.to(DEVICE)
    else:
        s2, hist, _ = train_autoencoder(s2, R_real, ReconLoss(w_mse=1.0, w_l1=0.5), epochs=s2ep, lr=1e-3,
                                        masks_u8=M, log_every=5, seed=args.seed)
        torch.save(s2.state_dict(), f); hist.to_csv(os.path.join(OUT, f"s2_{name}_history.csv"), index=False)
    s2.eval()
    lat[f"Z2_{name}_real"] = embed_and_reconstruct(s2, R_real, M)[0]
    for k, (A, AM) in Pin.items():
        lat[f"Z2_{name}_{k}"] = embed_and_reconstruct(s2, stage2_images(s1, A, AM, kind, gain), AM)[0]
    print(f"stage 2 {name} done | {time.time()-t0:.0f}s", flush=True)

np.savez(os.path.join(OUT, "latents.npz"), **lat)
json.dump(vars(args), open(os.path.join(OUT, "args.json"), "w"), indent=1)
print(f"saved {OUT} | total {time.time()-t0:.0f}s", flush=True)
