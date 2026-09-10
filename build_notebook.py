"""Assemble NSSC_DA_HiRISE.ipynb.

Code cells for library functions are pasted verbatim from nssc_core.py (by section
marker) so the notebook is self-contained and matches the code that produced the
checkpoints. Narrative markdown lives in notebook_text.py.
"""
import re
import nbformat as nbf
from notebook_text import T

core = open("nssc_core.py", encoding="utf-8").read()
SECTIONS = dict(re.findall(r"# === SECTION: (\w+) ===\n(.*?)(?=\n# === SECTION: |\Z)", core, re.S))

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
sec = lambda name: code(f"# ---- nssc_core.py :: {name}\n" + SECTIONS[name].strip())

# ------------------------------------------------------------------ 0. setup
md(T["title"])
md("## 0. Setup")
sec("setup")
code('''
%matplotlib inline
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 90, "font.size": 9})
ROOT = os.path.abspath(".")
DATA, OUT, CACHE = [os.path.join(ROOT, d) for d in ("data", "outputs", "cache")]
IMG_DIR = os.path.join(DATA, "images_unz", "images")      # images.zip extracted here
FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
RETRAIN = False     # True: retrain every version from scratch (~6 min each on an RTX 5070 Laptop GPU)
VERSIONS = ["v1", "v2", "v3", "v4"]
FINAL = "v3"        # version used for Phases 2-3 (v4 was tried and rejected - see changelog)
seed_everything(42)
print("torch", torch.__version__, "| device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")
def savefig(name): plt.savefig(os.path.join(FIG, name), dpi=130, bbox_inches="tight")
''')
sec("configs")

md("## 0.1 Data loading and sanity pass")
sec("data")
code('''
crop_index = pd.read_csv(os.path.join(DATA, "crop_metadata_index.csv"))
source_meta = pd.read_csv(os.path.join(DATA, "source_image_metadata.csv"))
files = [os.path.join(IMG_DIR, f) for f in crop_index.filename]
X = load_images(files, cache_path=os.path.join(CACHE, "images_u8.npy"))
assert len(X) == len(crop_index) and set(crop_index.source_image_id) == set(source_meta.source_image_id)
print(f"{len(X)} crops | {crop_index.source_image_id.nunique()} source observations | array {X.shape} {X.dtype}")
print(f"pixel range [{X.min()}, {X.max()}]  mean {X.mean():.1f}  -> rescaled to [0, 1] by /255 at batch time")
flat = X.reshape(len(X), -1)
nodata = (flat <= 2).mean(1)            # share of near-black map-projection no-data pixels
img_mean, img_std = flat.mean(1) / 255, flat.std(1) / 255
print(f"crops with a black no-data border (>1% of pixels): {(nodata > .01).mean():.1%}")
print(f"crops with >1% saturated-white pixels: {((flat >= 253).mean(1) > .01).sum()}")
''')
code('''
rng = np.random.default_rng(0); sel = rng.choice(len(X), 24, replace=False)
fig, ax = plt.subplots(3, 8, figsize=(16, 6.4))
for a, i in zip(ax.ravel(), sel):
    a.imshow(X[i], cmap="gray", vmin=0, vmax=255); a.set_title(crop_index.filename[i], fontsize=7); a.axis("off")
plt.suptitle("Fig 0.1 - Random batch of crops (note the black no-data wedges at strip edges)"); plt.tight_layout(); savefig("fig01_random_batch.png"); plt.show()
fig, ax = plt.subplots(1, 3, figsize=(14, 3))
ax[0].hist(img_mean, 80); ax[0].set_title("per-crop mean intensity")
ax[1].hist(img_std, 80); ax[1].set_title("per-crop std (contrast)")
ax[2].hist(nodata, 80, log=True); ax[2].set_title("per-crop no-data fraction (log)")
plt.tight_layout(); savefig("fig02_pixel_stats.png"); plt.show()
''')
md(T["sanity"])

md("## 0.2 No-data (map-projection) masking - introduced in v3")
md(T["nodata"])
sec("nodata")
code('''
M = compute_nodata_masks(X, cache_path=os.path.join(CACHE, "nodata_masks.npy"))
Xf = fill_nodata(X, M)
INPUT = {v: ((Xf, M) if CONFIGS[v].get("nodata") else (X, None)) for v in VERSIONS}   # model input per version
has_mask = M.reshape(len(M), -1).any(1)
near_black = (flat <= 5).mean(1)
print(f"crops with a masked no-data wedge: {has_mask.mean():.1%} (mean masked share {M[has_mask].mean():.1%})")
print(f"crops with >1% near-black pixels that are NOT masked (interior dark features): {((near_black > .01) & ~has_mask).sum()}")
sel = list(np.random.default_rng(0).choice(np.where(M.reshape(len(M), -1).mean(1) > .05)[0], 4, replace=False))
sel += list(np.where((near_black > .01) & ~has_mask)[0][:4])
fig, ax = plt.subplots(3, len(sel), figsize=(2.3 * len(sel), 7))
for j, i in enumerate(sel):
    ax[0, j].imshow(X[i], cmap="gray", vmin=0, vmax=255); ax[0, j].set_title(crop_index.filename[i], fontsize=8)
    ax[1, j].imshow(M[i], cmap="gray", vmin=0, vmax=1); ax[2, j].imshow(Xf[i], cmap="gray", vmin=0, vmax=255)
ax[0, 0].set_ylabel("original"); ax[1, 0].set_ylabel("no-data mask"); ax[2, 0].set_ylabel("v3 input (filled)")
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
plt.suptitle("Fig 0.3 - Wedges are masked; thin/interior dark features (right) are deliberately kept"); plt.tight_layout()
savefig("fig02b_nodata_masks.png"); plt.show()
''')

# ------------------------------------------------------------------ Phase 1
md("# Phase 1 - Deep Latent Compression (Autoencoders)")
md("## 1.1 Architecture Design")
md(T["arch"])
sec("model")
code('''
m = ConvAE(**CONFIGS[FINAL]["model"])
h, rows = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE), []
for blk in m.encoder:
    h = blk(h); rows.append(("encoder", type(blk).__name__, tuple(h.shape[1:])))
rows.append(("bottleneck", f"1x1 Conv -> 8x8x{m.spatial_latent}, flatten" if m.spatial_latent else "Linear", (m.latent_dim,)))
print(pd.DataFrame(rows, columns=["part", "block", "output shape"]).to_string())
print(f"latent: 1-D vector of {m.latent_dim} floats | parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")
''')
md("### 1.1.1 Latent dimensionality: fidelity vs separability sweep")
md(T["latent_sweep"])
code('''
sweep_path = os.path.join(OUT, "latent_sweep.csv")
summ = sorted(glob.glob(os.path.join(OUT, "sweep_*", "summary.json")))
if summ:   # written by run_version.py for the final config with d in {64,128,256,512}, 12-epoch schedule
    pd.DataFrame([json.load(open(p)) for p in summ]).sort_values("latent_dim").to_csv(sweep_path, index=False)
if os.path.exists(sweep_path):
    sweep = pd.read_csv(sweep_path); print(sweep.to_string(index=False))
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.2))
    ax[0].plot(sweep.latent_dim, sweep.val_ssim, "o-"); ax[0].set_title("val SSIM (fidelity) ↑")
    ax[1].plot(sweep.latent_dim, sweep.pca90, "o-"); ax[1].set_title("PCA dims for 90% latent variance")
    ax[2].plot(sweep.latent_dim, sweep.tail_ratio, "o-"); ax[2].set_title("novelty tail ratio (separability) ↑")
    for a in ax: a.set_xscale("log", base=2); a.set_xlabel("latent dim")
    plt.tight_layout(); savefig("fig03_latent_sweep.png"); plt.show()
''')
md(T.get("latent_choice", ""))

md("## 1.2 Loss Function Design")
md(T["loss"])
sec("losses")

md("### 1.2.1 Training (all versions v1-v4)")
sec("training")
code('''
def train_or_load(v):
    """Train version v (or load its checkpoint produced by exactly this code)."""
    cfg, out = CONFIGS[v], os.path.join(OUT, v)
    model = ConvAE(**cfg["model"])
    if RETRAIN or not os.path.exists(os.path.join(out, "model.pt")):
        os.makedirs(out, exist_ok=True)
        weights = None
        if cfg.get("trim_from"):   # v3: exclude crops the previous version flagged
            prev = np.load(os.path.join(OUT, cfg["trim_from"], "novelty.npy"))
            weights = (prev <= cfg["trim_threshold"]).astype(float)
        Xin, Min = INPUT[v]
        model, hist, _ = train_autoencoder(model, Xin, ReconLoss(**cfg["loss"]), epochs=cfg["epochs"],
                                           lr=cfg["lr"], sample_weights=weights, masks_u8=Min)
        torch.save(model.state_dict(), os.path.join(out, "model.pt")); hist.to_csv(os.path.join(out, "history.csv"), index=False)
    model.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location=DEVICE))
    return model.to(DEVICE).eval(), pd.read_csv(os.path.join(out, "history.csv"))

R = {}
for v in VERSIONS:
    model, hist = train_or_load(v)
    Z, mse, ssim = embed_and_reconstruct(model, *INPUT[v])   # v3: filled input, metrics on valid pixels
    R[v] = dict(model=model, hist=hist, Z=Z, mse=mse, ssim=ssim)
    print(f"{v}: final val MSE {hist.val_mse.iloc[-1]:.5f} | val SSIM {hist.val_ssim.iloc[-1]:.4f} | "
          f"epochs {len(hist)} | {hist.sec.sum()/60:.1f} min")
''')
code('''
fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
for v in VERSIONS:
    h = R[v]["hist"]
    ax[0].plot(h.epoch, h.train_loss, label=v); ax[1].plot(h.epoch, h.val_mse, label=v); ax[2].plot(h.epoch, h.val_ssim, label=v)
ax[0].set_title("train loss (each version's own objective)"); ax[0].set_yscale("log")
ax[1].set_title("held-out MSE ↓"); ax[2].set_title("held-out SSIM ↑")
for a in ax: a.legend(); a.set_xlabel("epoch")
plt.tight_layout(); savefig("fig04_training_curves.png"); plt.show()

show = [int(np.argmax(img_std * (nodata < .01)))]   # highest-texture crop without no-data
rng = np.random.default_rng(3); show += list(rng.choice(np.where(nodata < .01)[0], 3, replace=False)); show += [int(np.argmax(nodata))]
fig, ax = plt.subplots(len(show), 1 + len(VERSIONS), figsize=(2.6 * (1 + len(VERSIONS)), 2.6 * len(show)))
for r_, i in enumerate(show):
    ax[r_, 0].imshow(X[i], cmap="gray", vmin=0, vmax=255); ax[r_, 0].set_title(crop_index.filename[i], fontsize=8)
    for c_, v in enumerate(VERSIONS):
        _, rec = reconstruct(R[v]["model"], INPUT[v][0], [i])
        ax[r_, c_ + 1].imshow(rec[0], cmap="gray", vmin=0, vmax=1)
        ax[r_, c_ + 1].set_title(f"{v}  SSIM {R[v]['ssim'][i]:.3f}", fontsize=8)
for a in ax.ravel(): a.axis("off")
plt.suptitle("Fig 1.2 - Reconstructions across versions (fine texture is where MSE-only fails)"); plt.tight_layout()
savefig("fig05_recon_versions.png"); plt.show()
''')

md("## 1.3 Latent Space Visualisation")
md(T["tsne_intro"])
code('''
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
def tsne_embed(Z, seed=0):
    Zp = PCA(50, random_state=seed).fit_transform((Z - Z.mean(0)) / (Z.std(0) + 1e-8))
    return TSNE(2, perplexity=40, init="pca", random_state=seed).fit_transform(Zp)
tsne_path = os.path.join(OUT, FINAL, "tsne.npy")
E = np.load(tsne_path) if (os.path.exists(tsne_path) and not RETRAIN) else tsne_embed(R[FINAL]["Z"])
np.save(tsne_path, E)
res = crop_index.source_image_id.map(source_meta.set_index("source_image_id").resolution).values
fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
for a, c, t in [(ax[0], img_mean, "mean intensity"), (ax[1], img_std, "contrast (std)"),
                (ax[2], nodata, "no-data fraction"), (ax[3], res, "source resolution (m/px)")]:
    sc = a.scatter(E[:, 0], E[:, 1], c=c, s=2, cmap="viridis"); plt.colorbar(sc, ax=a); a.set_title(f"t-SNE ({FINAL}) coloured by {t}")
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); savefig("fig06_tsne_properties.png"); plt.show()
''')
code('''
# thumbnails sampled across the map to show what each region contains (visual interpretation only)
fig, ax = plt.subplots(figsize=(11, 11)); ax.scatter(E[:, 0], E[:, 1], s=1, c="lightgray")
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
gx = np.linspace(E[:, 0].min(), E[:, 0].max(), 7); gy = np.linspace(E[:, 1].min(), E[:, 1].max(), 7)
for x_ in gx:
    for y_ in gy:
        d = np.hypot(E[:, 0] - x_, E[:, 1] - y_); i = int(np.argmin(d))
        if d[i] < (gx[1] - gx[0]) * 0.4:
            ax.add_artist(AnnotationBbox(OffsetImage(X[i][::3, ::3], cmap="gray", zoom=1.0), E[i], frameon=False))
ax.set_title(f"Fig 1.3b - t-SNE of {FINAL} latents with nearest-crop thumbnails"); ax.set_xticks([]); ax.set_yticks([])
savefig("fig07_tsne_thumbs.png"); plt.show()
''')
md(T["tsne_read"])

# ------------------------------------------------------------------ Phase 2
md("# Phase 2 - Isolation Forest Novelty Engine")
md("## 2.1 Novelty Scoring")
md(T["if_intro"])
sec("iforest")
code('''
for v in VERSIONS:
    p = os.path.join(OUT, v, "novelty.npy")
    if os.path.exists(p) and not RETRAIN: R[v]["nov"] = np.load(p)
    else: R[v]["nov"] = novelty_scores(R[v]["Z"])[0]; np.save(p, R[v]["nov"])
nov = R[FINAL]["nov"]
# convention check: our score must be anti-correlated with sklearn's decision_function (higher = more normal)
chk = IsolationForest(n_estimators=200, random_state=0).fit(R[FINAL]["Z"])
print("corr(novelty, sklearn decision_function) =", round(np.corrcoef(nov, chk.decision_function(R[FINAL]["Z"]))[0, 1], 3), "(must be strongly negative)")
# seed stability of the ranking
runs = [ -IsolationForest(n_estimators=500, max_samples=256, random_state=s, n_jobs=-1).fit(R[FINAL]["Z"]).score_samples(R[FINAL]["Z"]) for s in (10, 11)]
print("Spearman rank corr between two independent forests:", round(pd.Series(runs[0]).corr(pd.Series(runs[1]), method="spearman"), 4))
print("Spearman(novelty, reconstruction MSE):", round(pd.Series(nov).corr(pd.Series(R[FINAL]["mse"]), method="spearman"), 3))
''')

md("## 2.2 Statistical Thresholding")
md(T["thr_intro"])
sec("threshold")
code('''
diag = distribution_diagnostics(nov)
print(pd.Series(diag).round(5).to_string())
t_mad = mad_threshold(nov, 3.5)
t_knee, i_knee = kneedle_threshold(nov)
t_gmm, gmm = gmm_threshold(nov)
t_ms3 = nov.mean() + 3 * nov.std()
cands = pd.DataFrame([
    ("Modified z-score (median + 3.5 MAD/0.6745)", t_mad),
    ("Kneedle elbow of sorted scores", t_knee),
    ("2-component GMM posterior = 0.5", t_gmm),
    ("mu + 3 sigma (assumes Gaussian - rejected)", t_ms3)], columns=["method", "threshold"])
cands["n_flagged"] = [(nov > t).sum() for t in cands.threshold]
print(cands.to_string(index=False))
''')
code('''
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
s_sorted = np.sort(nov)
ax[0].plot(s_sorted, lw=1.2, c="k")
for t, c, l in [(t_mad, "r", "MAD 3.5"), (t_knee, "g", "Kneedle"), (t_gmm, "m", "GMM"), (t_ms3, "b", "μ+3σ")]:
    ax[0].axhline(t, ls="--", c=c, label=f"{l}: {t:.3f} ({(nov > t).sum()})")
ax[0].set_title("sorted Novelty Score"); ax[0].set_xlabel("rank"); ax[0].legend(fontsize=8)
ax[1].hist(nov, 150, color="0.5"); ax[1].set_yscale("log"); ax[1].axvline(THRESHOLD if "THRESHOLD" in globals() else t_mad, c="r")
ax[1].set_title("Novelty Score histogram (log count)")
stats.probplot(nov, dist="norm", plot=ax[2]); ax[2].set_title("Normal Q-Q plot of Novelty Scores")
plt.tight_layout(); savefig("fig08_threshold_candidates.png"); plt.show()
''')
md(T["thr_choice"])
code(T["thr_code"])
code('''
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
sc = ax[0].scatter(E[:, 0], E[:, 1], c=nov, s=2, cmap="magma"); plt.colorbar(sc, ax=ax[0], label="Novelty Score")
ax[0].set_title(f"t-SNE ({FINAL}) coloured by Novelty Score")
ax[1].scatter(E[~flagged, 0], E[~flagged, 1], s=2, c="0.8", label="normal")
ax[1].scatter(E[flagged, 0], E[flagged, 1], s=10, c="r", label=f"flagged ({flagged.sum()})"); ax[1].legend()
ax[1].set_title("crops above the statistical threshold")
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); savefig("fig09_tsne_flagged.png"); plt.show()
''')

md("### Contamination parameter")
md(T["contamination"])

md("## 2.3 Image Location Analysis")
md(T["loc_intro"])
code('''
scored = crop_index.assign(novelty=nov, flagged=flagged, recon_mse=R[FINAL]["mse"], nodata=nodata) \\
                   .merge(source_meta, on="source_image_id", how="left", validate="many_to_one")
assert scored.latitude.notna().all()
scored.to_csv(os.path.join(OUT, "novelty_scores_with_metadata.csv"), index=False)
print(scored.sort_values("novelty", ascending=False).head(15).to_string(index=False))
''')
code('''
per_src = scored.groupby("source_image_id").agg(n=("flagged", "size"), n_flag=("flagged", "sum"), mean_nov=("novelty", "mean"),
                                                 latitude=("latitude", "first"), longitude=("longitude", "first"),
                                                 sun_angle=("sun_angle", "first"), season=("season", "first"), resolution=("resolution", "first"))
per_src["rate"] = per_src.n_flag / per_src.n
base = scored.flagged.mean()
from scipy.stats import fisher_exact, binomtest
per_src["p_enrich"] = [binomtest(int(k), int(n), base, alternative="greater").pvalue for k, n in zip(per_src.n_flag, per_src.n)]
m_ = len(per_src); per_src["p_bonf"] = (per_src.p_enrich * m_).clip(upper=1)
print(f"base flag rate {base:.2%}; sources with >=1 flagged crop: {(per_src.n_flag > 0).sum()} / {m_}")
print(per_src.sort_values("n_flag", ascending=False).head(12).round(4).to_string())
''')
code('''
fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
# the provided metadata is quantised: latitude on a 5° grid (plus the poles), longitude only 0° or 180°E
lat_rate = scored.groupby(["latitude", "longitude"]).flagged.mean().unstack()
for lon, c in zip(lat_rate.columns, ["tab:blue", "tab:orange"]):
    ax[0].plot(lat_rate.index, lat_rate[lon] * 100, "o-", c=c, label=f"lon {lon:.0f}°E")
ax[0].axhline(scored.flagged.mean() * 100, c="k", ls=":", label="overall flag rate")
ax[0].set_xlabel("latitude band (°)"); ax[0].set_ylabel("% of crops flagged"); ax[0].legend(fontsize=8)
ax[0].set_title("flag rate by latitude band and longitude")
for fl, c in [(False, "0.6"), (True, "r")]:
    d = scored[scored.flagged == fl]; ax[1].scatter(d.sun_angle, d.novelty, s=3 if not fl else 14, c=c, label="flagged" if fl else "normal")
ax[1].axhline(THRESHOLD, c="r", ls="--"); ax[1].set_xlabel("sun angle (°)"); ax[1].set_ylabel("Novelty Score"); ax[1].legend()
order_s = ["N-spring", "N-summer", "N-autumn", "N-winter"]
ax[2].boxplot([scored.novelty[scored.season == s] for s in order_s], labels=order_s, showfliers=True, flierprops=dict(ms=2))
ax[2].axhline(THRESHOLD, c="r", ls="--"); ax[2].set_title("Novelty Score by season")
plt.tight_layout(); savefig("fig10_location.png"); plt.show()
from scipy.stats import spearmanr, kruskal, chi2_contingency
LOC = dict(
    rho_sun=spearmanr(scored.sun_angle, scored.novelty), rho_abslat=spearmanr(scored.latitude.abs(), scored.novelty),
    p_season=kruskal(*[scored.novelty[scored.season == s] for s in order_s]).pvalue,
    p_resolution=kruskal(*[g.novelty for _, g in scored.groupby("resolution")]).pvalue,
    p_chi2_season=chi2_contingency(pd.crosstab(scored.season, scored.flagged))[1],
    n_src_flagged=int((per_src.n_flag > 0).sum()), n_src_enriched=int((per_src.p_bonf < .05).sum()))
LOC = {k: ([float(v[0]), float(v[1])] if isinstance(v, tuple) or hasattr(v, "pvalue") else float(v)) for k, v in LOC.items()}
for k, v in LOC.items(): print(f"{k:16s} {v}")
''')
md(T["loc_read"])

md("## 2.4 Optional Metadata Fusion (+2)")
md(T["fusion_intro"])
code('''
Zs = (R[FINAL]["Z"] - R[FINAL]["Z"].mean(0)) / (R[FINAL]["Z"].std(0) + 1e-8)
meta_vec = np.column_stack([
    (scored.sun_angle - scored.sun_angle.mean()) / scored.sun_angle.std(),
    np.sin(np.radians(scored.latitude)), np.cos(np.radians(scored.latitude)),
    np.sin(np.radians(scored.longitude)), np.cos(np.radians(scored.longitude)),
    np.log2(scored.resolution / 0.25),
    pd.get_dummies(scored.season).values.astype(float)])
fused = np.hstack([Zs, meta_vec])
nov_fused = novelty_scores(fused)[0]
thr_fused = mad_threshold(nov_fused, 3.5)   # same statistical rule as the image-only pipeline
flag_fused = nov_fused > thr_fused
a, b = set(np.where(flagged)[0]), set(np.where(flag_fused)[0])
FUS = dict(n_image=len(a), n_fused=len(b), overlap=len(a & b), jaccard=len(a & b) / len(a | b),
           rank_corr=float(pd.Series(nov).corr(pd.Series(nov_fused), method="spearman")),
           added=crop_index.filename[sorted(b - a)].tolist(), dropped=crop_index.filename[sorted(a - b)].tolist())
for k, v in FUS.items(): print(f"{k:10s} {v if not isinstance(v, list) else v[:15]}")
''')
md(T["fusion_read"])

# ------------------------------------------------------------------ Phase 3
md("# Phase 3 - Reconstruction Interpretability")
md("## 3.1 Pixel-Wise Error Heatmaps")
md(T["hm_intro"])
sec("heatmaps")
code('''
flag_idx = np.where(flagged)[0]
top5 = flag_idx[np.argsort(-nov[flag_idx])][:5]      # top-5 *within* the thresholded set
Xin, Min = INPUT[FINAL]                                # v3 decoder input: no-data filled
orig, rec = reconstruct(R[FINAL]["model"], Xin, top5)
valid = np.ones_like(orig) if Min is None else 1.0 - Min[top5]
fig, ax = plt.subplots(len(top5), 4, figsize=(15, 3.8 * len(top5)), squeeze=False)
desc = []
for r_, i in enumerate(top5):
    e = error_map(orig[r_], rec[r_], valid=valid[r_])  # no-data pixels carry no error by construction
    desc.append(dict(rank=r_ + 1, filename=crop_index.filename[i], source=crop_index.source_image_id[i],
                     novelty=nov[i], recon_mse=R[FINAL]["mse"][i], **heatmap_descriptors(orig[r_], e, valid[r_])))
    ax[r_, 0].imshow(X[i], cmap="gray", vmin=0, vmax=255); ax[r_, 0].set_title(f"#{r_+1} {crop_index.filename[i]}  novelty {nov[i]:.3f}")
    ax[r_, 1].imshow(rec[r_], cmap="gray", vmin=0, vmax=1); ax[r_, 1].set_title(f"{FINAL} reconstruction")
    im = ax[r_, 2].imshow(e, cmap="inferno"); plt.colorbar(im, ax=ax[r_, 2], fraction=.046); ax[r_, 2].set_title("squared error (σ=2 smoothed)")
    ax[r_, 3].imshow(X[i], cmap="gray", vmin=0, vmax=255); ax[r_, 3].imshow(e, cmap="inferno", alpha=.55)
    ax[r_, 3].set_title("overlay")
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); savefig("fig11_heatmaps_top5.png"); plt.show()
desc = pd.DataFrame(desc); print(desc.round(4).to_string(index=False))
''')
code('''
# reference: the same descriptors on 500 random un-flagged crops, to say what "unusual" means
rng = np.random.default_rng(0); ref_idx = rng.choice(np.where(~flagged)[0], 500, replace=False)
Xin, Min = INPUT[FINAL]
o_, r_ = reconstruct(R[FINAL]["model"], Xin, ref_idx)
v_ = np.ones_like(o_) if Min is None else 1.0 - Min[ref_idx]
ref = pd.DataFrame([heatmap_descriptors(o_[k], error_map(o_[k], r_[k], valid=v_[k]), v_[k]) for k in range(len(ref_idx))])
pct = desc[ref.columns].apply(lambda col: [(ref[col.name] < v).mean() * 100 for v in col])
pct.insert(0, "filename", desc.filename)
print("percentile of each top-5 descriptor relative to 500 normal crops:"); print(pct.round(0).to_string(index=False))
''')
md("### 3.1b Supplementary heatmaps: visually noticed crops (flagged and missed)")
code('''
supp = {9828: "Pentagon (terrestrial)", 6476: "periodic grid", 5395: "bright + thin line",
        15: "black bar", 503: "two streaks", 676: "dead-pixel blocks"}
idx_s = np.array([k - 1 for k in supp])
Xin, Min = INPUT[FINAL]
o_s, r_s = reconstruct(R[FINAL]["model"], Xin, idx_s)
v_s = np.ones_like(o_s) if Min is None else 1.0 - Min[idx_s]
rank_all = (-nov).argsort().argsort() + 1
fig, ax = plt.subplots(2, len(idx_s), figsize=(2.8 * len(idx_s), 6.2))
for j, (i, lab) in enumerate(zip(idx_s, supp.values())):
    e = error_map(o_s[j], r_s[j], valid=v_s[j])
    mse_pct = (R[FINAL]["mse"] < R[FINAL]["mse"][i]).mean() * 100
    ax[0, j].imshow(X[i], cmap="gray", vmin=0, vmax=255)
    ax[0, j].set_title(f"{lab}\\n{crop_index.filename[i]} | rank {rank_all[i]} | {'FLAGGED' if flagged[i] else 'not flagged'}", fontsize=8)
    ax[1, j].imshow(X[i], cmap="gray", vmin=0, vmax=255); ax[1, j].imshow(e, cmap="inferno", alpha=.6)
    ax[1, j].set_title(f"recon MSE {R[FINAL]['mse'][i]:.4f} ({mse_pct:.0f}th pct)", fontsize=8)
for a in ax.ravel(): a.axis("off")
plt.suptitle("Fig 3.1b - Decoder residual on visually noticed crops (diagnostic only)"); plt.tight_layout()
savefig("fig11b_supplementary_heatmaps.png"); plt.show()
''')

md("## 3.2 Geological Report")
md(T["geo_report"])

# ------------------------------------------------------------------ Phase 4
md("# Phase 4 - Architecture Iteration & Design Journal")
md(T["changelog_intro"])
code('''
rows = []
for v in VERSIONS:
    h, n_ = R[v]["hist"], R[v]["nov"]
    thr_v = mad_threshold(n_, 3.5)
    Zv = R[v]["Z"]; pv = np.linalg.svd(Zv - Zv.mean(0), compute_uv=False) ** 2; pv /= pv.sum()
    rows.append(dict(version=v, val_mse=h.val_mse.iloc[-1], val_ssim=h.val_ssim.iloc[-1],
                     nodata_mse_ratio=R[v]["mse"][nodata > .1].mean() / R[v]["mse"][nodata < .01].mean(),
                     pca90=int(np.searchsorted(np.cumsum(pv), .9) + 1),
                     corr_nov_nodata=pd.Series(n_).corr(pd.Series(nodata), method="spearman"),
                     n_flag_mad35=int((n_ > thr_v).sum()),
                     flagged_with_nodata=float((nodata[n_ > thr_v] > .01).mean())))
cmp = pd.DataFrame(rows); print(cmp.round(4).to_string(index=False))
''')
code('''
fig, ax = plt.subplots(1, len(VERSIONS), figsize=(16, 3.2), sharey=True)
for a, v in zip(ax, VERSIONS):
    n_ = R[v]["nov"]; b_ = nodata > .01
    a.hist([n_[~b_], n_[b_]], 100, stacked=True, color=["0.55", "tab:orange"], label=["no border", "has no-data border"], log=True)
    a.axvline(mad_threshold(n_, 3.5), c="r", ls="--"); a.set_title(f"{v} Novelty Score"); a.legend(fontsize=7)
plt.tight_layout(); savefig("fig12_versions_hist.png"); plt.show()
fig, ax = plt.subplots(len(VERSIONS), 8, figsize=(16, 2.2 * len(VERSIONS)))
for r_, v in enumerate(VERSIONS):
    for c_, i in enumerate(np.argsort(-R[v]["nov"])[:8]):
        ax[r_, c_].imshow(X[i], cmap="gray", vmin=0, vmax=255); ax[r_, c_].set_title(f"{v} #{c_+1} {crop_index.filename[i][7:12]}", fontsize=8)
for a in ax.ravel(): a.axis("off")
plt.suptitle("Fig 4.2 - Eight highest Novelty Scores per version"); plt.tight_layout(); savefig("fig13_versions_top8.png"); plt.show()
''')
code('''
# Spot checks: crops we noticed by eye (sanity pass / earlier versions). Used only to diagnose
# failure modes in the changelog - never to choose the threshold.
SPOT = {15: "black bar", 503: "two streaks", 676: "dead-pixel blocks", 9828: "Pentagon (terrestrial)",
        5508: "terrestrial buildings", 6476: "periodic grid", 7971: "periodic grid", 5395: "bright + thin line",
        11: "dark terrain (control)", 385: "dark terrain (control)"}
spot = pd.DataFrame({"note": list(SPOT.values())}, index=[f"sample_{k:05d}.jpg" for k in SPOT])
for v in VERSIONS:
    n_ = R[v]["nov"]; rank = (-n_).argsort().argsort() + 1; thr_v = mad_threshold(n_, 3.5)
    spot[f"{v} rank"] = [int(rank[k - 1]) for k in SPOT]
    spot[f"{v} flagged"] = [bool(n_[k - 1] > thr_v) for k in SPOT]
print(spot.to_string())
''')
md(open("CHANGELOG.md", encoding="utf-8").read().split("\n", 1)[1])   # the journal, verbatim

md("# Final answers")
md(T["final"])
code('''
flag_table = scored[scored.flagged].sort_values("novelty", ascending=False)[
    ["filename", "source_image_id", "novelty", "recon_mse", "latitude", "longitude", "sun_angle", "season", "resolution"]]
flag_table.to_csv(os.path.join(OUT, "flagged_anomalies.csv"), index=False)
print(f"{len(flag_table)} crops flagged (threshold {THRESHOLD:.4f}); full list -> outputs/flagged_anomalies.csv")
print(flag_table.head(30).round(4).to_string(index=False))
results = dict(n_crops=len(X), n_sources=int(crop_index.source_image_id.nunique()), final_version=FINAL,
               final_config=CONFIGS[FINAL], threshold=float(THRESHOLD), n_flagged=int(flagged.sum()),
               diag={k: float(v) for k, v in diag.items()}, candidates=cands.to_dict("records"),
               top5=desc.to_dict("records"), top5_pct=pct.to_dict("records"), versions=cmp.to_dict("records"),
               location=LOC, per_src_top=per_src.sort_values("n_flag", ascending=False).head(10).reset_index().to_dict("records"),
               fusion=FUS, flagged=flag_table.round(5).to_dict("records"),
               spot=spot.reset_index().rename(columns={"index": "crop"}).to_dict("records"),
               sweep=pd.read_csv(sweep_path).to_dict("records") if os.path.exists(sweep_path) else None)
json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "NSSC_DA_HiRISE.ipynb")
print("wrote NSSC_DA_HiRISE.ipynb with", len(cells), "cells")
