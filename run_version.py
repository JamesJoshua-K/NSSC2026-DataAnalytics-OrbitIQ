"""Train one model version and dump every artefact the notebook / changelog needs.

usage: python run_version.py v1 [--epochs N]
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nssc_core import *

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--latent", type=int, default=None)
    args = ap.parse_args()
    cfg = json.loads(json.dumps(CONFIGS[args.version]))
    if args.epochs: cfg["epochs"] = args.epochs
    if args.latent:
        if cfg["model"].get("spatial_latent"):
            cfg["model"]["spatial_latent"] = args.latent // 64   # 8x8 grid x C channels
        cfg["model"]["latent_dim"] = args.latent
    name = args.tag or args.version
    out = os.path.join(ROOT, "outputs", name)
    os.makedirs(out, exist_ok=True)

    idx = pd.read_csv(os.path.join(ROOT, "data/crop_metadata_index.csv"))
    X = np.load(os.path.join(ROOT, "cache/images_u8.npy"))
    weights = None   # uniform sampling of all crops (no data-cleaning of any kind)

    pcfg = CONFIGS[cfg["residual_of"]] if cfg.get("residual_of") else cfg   # stage 1 sees exactly its own input
    X, M = preprocess(pcfg, X, mask_cache=os.path.join(ROOT, "cache/nodata_masks.npy"))
    if M is not None:
        print("no-data masks: %.1f%% of crops have masked pixels" % (100 * M.reshape(len(M), -1).any(1).mean()))
    Z1 = None
    if cfg.get("residual_of"):   # two-stage: stage 2 is trained on the residual of the trained stage-1 model
        stage1 = load_model(cfg["residual_of"], os.path.join(ROOT, "outputs"))
        Z1, mse1, ssim1 = embed_and_reconstruct(stage1, X, M)
        X = residual_images(stage1, X, M)
        print("stage 2 trains on the reconstruction residuals of", cfg["residual_of"])

    seed_everything(42)   # identical initial weights on every run
    model = ConvAE(**cfg["model"])
    print(name, cfg, "params %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    model, hist, val_idx = train_autoencoder(model, X, ReconLoss(**cfg["loss"]), epochs=cfg["epochs"],
                                             lr=cfg["lr"], sample_weights=weights, masks_u8=M)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    hist.to_csv(os.path.join(out, "history.csv"), index=False)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)

    Z, mse, ssim = embed_and_reconstruct(model, X, M)
    if Z1 is not None:
        Z, mse, ssim = np.hstack([Z1, Z]), mse1, ssim1   # final latent = [scene code | residual code]
    np.save(os.path.join(out, "latents.npy"), Z)
    np.save(os.path.join(out, "recon_mse.npy"), mse)
    np.save(os.path.join(out, "recon_ssim.npy"), ssim)
    t0 = time.time()
    nov, nov_sd = novelty_scores(Z)
    np.save(os.path.join(out, "novelty.npy"), nov)
    print("IF scored in %.1fs" % (time.time() - t0))

    # ---- diagnostics
    var = Z.var(0)
    print("latent var: min %.4g median %.4g max %.4g | dims with var<1e-3 of max: %d"
          % (var.min(), np.median(var), var.max(), int((var < 1e-3 * var.max()).sum())))
    pca_var = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2
    pca_var /= pca_var.sum()
    print("PCA: dims for 90%% var = %d, 99%% = %d" % (np.searchsorted(np.cumsum(pca_var), .9) + 1,
                                                   np.searchsorted(np.cumsum(pca_var), .99) + 1))
    d = distribution_diagnostics(nov)
    print("novelty:", {k: round(float(v), 4) for k, v in d.items()})
    t_mad = mad_threshold(nov); t_knee, _ = kneedle_threshold(nov); t_gmm, _ = gmm_threshold(nov)
    t_ms = nov.mean() + 3 * nov.std()
    for nm, t in [("MAD3.5", t_mad), ("kneedle", t_knee), ("GMM", t_gmm), ("mu+3sd", t_ms)]:
        print(f"  thr {nm:8s} {t:.4f} -> flagged {(nov > t).sum()}")
    print("seed-std of novelty: median %.4f" % np.median(nov_sd))
    nodata = (np.load(os.path.join(ROOT, "cache/images_u8.npy")) <= 2).mean((1, 2))
    summary = dict(version=name, latent_dim=cfg["model"]["latent_dim"], epochs=cfg["epochs"],
                   val_mse=float(hist.val_mse.iloc[-1]), val_ssim=float(hist.val_ssim.iloc[-1]),
                   pca90=int(np.searchsorted(np.cumsum(pca_var), .9) + 1),
                   tail_ratio=float((np.percentile(nov, 99.9) - d["median"]) / d["mad"]),
                   n_flag_mad35=int((nov > t_mad).sum()),
                   corr_nov_nodata=float(pd.Series(nov).corr(pd.Series(nodata), method="spearman")),
                   flagged_with_nodata=float((nodata[nov > t_mad] > .01).mean()))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("summary", summary)
    print("corr(novelty, recon_mse) spearman %.3f" % pd.Series(nov).corr(pd.Series(mse), method="spearman"))
    order = np.argsort(-nov)
    top = idx.iloc[order[:25]].assign(novelty=nov[order[:25]], mse=mse[order[:25]], ssim=ssim[order[:25]])
    print(top.to_string())
    top.to_csv(os.path.join(out, "top25.csv"), index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(hist.epoch, hist.train_loss, label="train loss"); ax[0].set_yscale("log"); ax[0].legend()
    ax[1].plot(hist.epoch, hist.val_mse, label="val MSE"); ax[1].legend()
    ax[2].plot(hist.epoch, hist.val_ssim, label="val SSIM"); ax[2].legend()
    plt.tight_layout(); plt.savefig(os.path.join(out, "curves.png"), dpi=70); plt.close()

    rng = np.random.default_rng(1)
    sel = np.concatenate([rng.choice(len(X), 6, replace=False), order[:6]])
    o, r = reconstruct(model, X, sel)
    fig, ax = plt.subplots(3, 12, figsize=(24, 6.5))
    for j, i in enumerate(sel):
        ax[0, j].imshow(o[j], cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(("rand " if j < 6 else "TOP ") + idx.filename[i][7:12], fontsize=9)
        ax[1, j].imshow(r[j], cmap="gray", vmin=0, vmax=1)
        ax[2, j].imshow(error_map(o[j], r[j]), cmap="inferno")
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(out, "recon_grid.png"), dpi=60); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    s = np.sort(nov)
    ax[0].plot(s); [ax[0].axhline(t, ls="--", c=c, label=l) for t, c, l in
                   [(t_mad, "r", "MAD3.5"), (t_knee, "g", "knee"), (t_gmm, "m", "GMM"), (t_ms, "k", "mu+3sd")]]
    ax[0].legend(); ax[0].set_title("sorted novelty")
    ax[1].hist(nov, 150, log=True); ax[1].set_title("novelty histogram (log)")
    plt.tight_layout(); plt.savefig(os.path.join(out, "scores.png"), dpi=70); plt.close()
    print("done", out)


if __name__ == "__main__":
    main()
