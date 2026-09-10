"""NSSC 2026 Data Analytics - Mars HiRISE unsupervised anomaly detection.

Core library: every function used by the notebook lives here, grouped by
problem-statement section. The notebook builder pastes each section verbatim
under its matching markdown heading, so this file is the single source of
truth for the code that appears in the submitted .ipynb.
"""

# === SECTION: setup ===
import os, glob, json, math, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage, stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 227
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
# bf16 autocast only where the GPU supports it natively (Ampere+); fp32 elsewhere, e.g. a Colab T4
USE_BF16 = DEVICE.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # reproducible up to residual GPU non-determinism
    torch.backends.cudnn.benchmark = False


# === SECTION: configs ===
# Every model version documented in the changelog (Phase 4). Each later version changes one
# thing motivated by an observed failure of the previous one - see CHANGELOG.md.
CONFIGS = {
    "v1": dict(model=dict(latent_dim=256, up="transpose", res_blocks=0, variational=False),
               loss=dict(w_mse=1.0), epochs=30, lr=1e-3),
    # v2: fix MSE blur -> structural (SSIM) + robust intensity (L1) + edge (gradient) loss
    "v2": dict(model=dict(latent_dim=256, up="transpose", res_blocks=0, variational=False),
               loss=dict(w_mse=0.0, w_l1=0.16, w_ssim=0.84, w_grad=0.1), epochs=30, lr=1e-3),
    # v3: remove the no-data nuisance -> mask map-projection wedges, fill with crop median,
    #     exclude them from the loss (and from reconstruction-error maps)
    "v3": dict(model=dict(latent_dim=256, up="transpose", res_blocks=0, variational=False),
               loss=dict(w_mse=0.0, w_l1=0.16, w_ssim=0.84, w_grad=0.1), epochs=30, lr=1e-3,
               nodata=True),
    # v4: keep local defects visible to the Isolation Forest -> spatial 8x8x4 bottleneck
    #     (1x1 conv instead of a global Linear), same 256-d 1-D latent, same loss and masking as v3
    "v4": dict(model=dict(latent_dim=256, up="transpose", res_blocks=0, variational=False, spatial_latent=4),
               loss=dict(w_mse=0.0, w_l1=0.16, w_ssim=0.84, w_grad=0.1), epochs=30, lr=1e-3,
               nodata=True),
    # v5: remove v3's brightness bias -> per-crop photometric normalisation (on top of v3)
    "v5": dict(model=dict(latent_dim=256, up="transpose", res_blocks=0, variational=False),
               loss=dict(w_mse=0.0, w_l1=0.16, w_ssim=0.84, w_grad=0.1), epochs=30, lr=1e-3,
               nodata=True, standardize=True),
    # v6: two-stage autoencoder -> stage 2 compresses stage 1's reconstruction residual into a
    #     64-d code; final latent = [256-d scene code | 64-d residual code] (320-d, 1-D)
    "v6": dict(model=dict(latent_dim=64, up="transpose", res_blocks=0, variational=False),
               loss=dict(w_mse=1.0, w_l1=0.5), epochs=30, lr=1e-3, nodata=True, residual_of="v3"),
}


# === SECTION: data ===
def list_images(root):
    files = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
             if p.lower().endswith(IMG_EXTS)]
    return sorted(files)


def load_images(files, cache_path=None):
    """Load every crop as uint8 [N, 227, 227]. Cached to .npy after first load."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    arr = np.empty((len(files), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i, f in enumerate(files):
        im = Image.open(f).convert("L")
        if im.size != (IMG_SIZE, IMG_SIZE):  # dataset is standardised; guard only
            im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr[i] = np.asarray(im)
    if cache_path:
        np.save(cache_path, arr)
    return arr


# === SECTION: nodata ===
def nodata_mask(img_u8, thr=5, min_area=0.005, open_px=5, dilate_px=2):
    """Map-projection no-data of one crop (True = no-data).

    A pixel is no-data if it belongs to a near-black (<= thr) connected region that
    (a) survives a 5x5 morphological opening - so thin dark lines, e.g. a dead sensor
        column that could itself be an anomaly, are NOT masked,
    (b) touches the crop border - strip edges enter from outside the crop, and
    (c) covers >= 0.5% of the crop.
    The mask is dilated by 2 px to cover JPEG ringing along the wedge edge.
    """
    m = ndimage.binary_opening(img_u8 <= thr, structure=np.ones((open_px, open_px)))
    if not m.any():
        return m
    lab, n = ndimage.label(m)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    sizes = ndimage.sum(m, lab, index=np.arange(n + 1))
    keep = np.zeros(n + 1, bool)
    keep[edge] = True
    keep[0] = False
    keep &= sizes >= min_area * m.size
    out = keep[lab]
    return ndimage.binary_dilation(out, iterations=dilate_px) if out.any() else out


def compute_nodata_masks(X, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    M = np.stack([nodata_mask(x) for x in X]).astype(np.uint8)
    if cache_path:
        np.save(cache_path, M)
    return M


def fill_nodata(X, M):
    """Replace no-data pixels by the median of the crop's valid pixels (a flat, texture-free fill)."""
    Xf = X.copy()
    for i in np.where(M.reshape(len(M), -1).any(1))[0]:
        valid = X[i][M[i] == 0]
        Xf[i][M[i] == 1] = np.uint8(np.median(valid)) if valid.size else 0
    return Xf


def standardize_images(X, M=None, target_mean=0.5, target_std=0.15):
    """Per-crop photometric normalisation: map each crop's valid pixels to a common mean and
    contrast, so the latent describes structure rather than albedo / exposure / sun angle.
    Pixels far from the crop's own statistics (a black bar, a hot line) stay extreme: they are
    clipped at 0 / 255, not removed."""
    Xs = np.empty_like(X)
    for i in range(len(X)):
        x = X[i].astype(np.float32)
        v = x[M[i] == 0] if M is not None else x.ravel()
        mu, sd = v.mean(), max(float(v.std()), 2.0)
        Xs[i] = np.clip(np.round(255 * (target_mean + target_std * (x - mu) / sd)), 0, 255).astype(np.uint8)
    return Xs


def preprocess(cfg, X, mask_cache=None):
    """Model input for a version: no-data masking + fill (configs with nodata=True) and
    per-crop photometric normalisation (standardize=True). Returns (images_u8, masks_u8 or None)."""
    M = None
    if cfg.get("nodata"):
        M = compute_nodata_masks(X, mask_cache)
        X = fill_nodata(X, M)
    if cfg.get("standardize"):
        X = standardize_images(X, M)
    return X, M


# === SECTION: model ===
def enc_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 2, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))


def dec_block(cin, cout, up):
    if up == "transpose":
        layer = nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False)
    else:  # nearest-neighbour upsample + conv: avoids checkerboard artefacts
        layer = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                              nn.Conv2d(cin, cout, 3, 1, 1, bias=False))
    return nn.Sequential(layer, nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c),
                                  nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.leaky_relu(x + self.body(x), 0.2)


class ConvAE(nn.Module):
    """Convolutional (optionally variational) autoencoder for 1x227x227 crops.

    Encoder: 5 stride-2 convs, 227 -> 114 -> 57 -> 29 -> 15 -> 8, then a linear
    bottleneck to a 1-D latent vector of size `latent_dim`.
    Decoder: linear -> 8x8 feature map, 5 x2 upsampling stages -> 256x256,
    centre-cropped back to 227x227, sigmoid output in [0, 1].
    """

    def __init__(self, latent_dim=256, chans=(32, 64, 128, 256, 256), up="transpose",
                 res_blocks=0, variational=False, spatial_latent=0):
        super().__init__()
        self.latent_dim, self.variational = latent_dim, variational
        enc, c_prev = [], 1
        for c in chans:
            enc.append(enc_block(c_prev, c))
            enc += [ResBlock(c) for _ in range(res_blocks)]
            c_prev = c
        self.encoder = nn.Sequential(*enc)
        self.c_last = chans[-1]
        flat = self.c_last * 8 * 8
        self.spatial_latent = spatial_latent
        if spatial_latent:
            # v4: 1x1-conv bottleneck -> 8x8 grid x C channels, flattened to a 1-D vector of 64*C.
            # Each cell summarises a ~63 px receptive field, so a local defect moves a few
            # coordinates strongly instead of being averaged away by a global linear layer.
            self.latent_dim = 64 * spatial_latent
            self.fc_mu = nn.Conv2d(self.c_last, spatial_latent, 1)
            self.fc_logvar = nn.Conv2d(self.c_last, spatial_latent, 1) if variational else None
            self.fc_dec = nn.Conv2d(spatial_latent, self.c_last, 1)
        else:
            self.fc_mu = nn.Linear(flat, latent_dim)
            self.fc_logvar = nn.Linear(flat, latent_dim) if variational else None
            self.fc_dec = nn.Linear(latent_dim, flat)
        dec, rev = [], list(chans[::-1]) + [16]
        for i in range(len(chans)):
            dec += [ResBlock(rev[i]) for _ in range(res_blocks)]
            dec.append(dec_block(rev[i], rev[i + 1], up))
        self.decoder = nn.Sequential(*dec)
        self.head = nn.Conv2d(16, 1, 3, 1, 1)

    def encode(self, x):
        h = self.encoder(x)
        if not self.spatial_latent:
            h = h.flatten(1)
        mu = self.fc_mu(h).flatten(1)                         # always a 1-D vector per image
        logvar = self.fc_logvar(h).flatten(1).clamp(-10, 10) if self.variational else None
        return mu, logvar

    def decode(self, z):
        if self.spatial_latent:
            h = self.fc_dec(z.view(-1, self.spatial_latent, 8, 8))
        else:
            h = self.fc_dec(z).view(-1, self.c_last, 8, 8)
        out = self.head(self.decoder(h))              # 256 x 256
        o = (out.shape[-1] - IMG_SIZE) // 2
        return torch.sigmoid(out[..., o:o + IMG_SIZE, o:o + IMG_SIZE])

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu
        if self.variational and self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


# === SECTION: losses ===
def _gauss_window(size=11, sigma=1.5, device="cpu"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim_map(x, y, window=None, C1=0.01 ** 2, C2=0.03 ** 2):
    """Differentiable per-pixel SSIM (Wang et al. 2004) for images in [0, 1]."""
    if window is None:
        window = _gauss_window(device=x.device)
    p = window.shape[-1] // 2
    mu_x, mu_y = F.conv2d(x, window, padding=p), F.conv2d(y, window, padding=p)
    sxx = F.conv2d(x * x, window, padding=p) - mu_x ** 2
    syy = F.conv2d(y * y, window, padding=p) - mu_y ** 2
    sxy = F.conv2d(x * y, window, padding=p) - mu_x * mu_y
    return ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))


def _wmean(t, w):
    """Mean of t over pixels, weighted by validity map w (None = plain mean)."""
    return t.mean() if w is None else (t * w).sum() / w.sum().clamp_min(1.0)


def gradient_loss(x, y, w=None):
    """L1 distance between horizontal/vertical finite differences (edge fidelity)."""
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    wx = None if w is None else w[..., :, 1:] * w[..., :, :-1]
    wy = None if w is None else w[..., 1:, :] * w[..., :-1, :]
    return _wmean((dx(x) - dx(y)).abs(), wx) + _wmean((dy(x) - dy(y)).abs(), wy)


class ReconLoss(nn.Module):
    """w_mse*MSE + w_l1*L1 + w_ssim*(1-SSIM) + w_grad*GradL1 (+ beta*KL for a VAE).

    Optional validity map w (1 = real terrain, 0 = no-data) restricts every term to valid pixels.
    """

    def __init__(self, w_mse=1.0, w_l1=0.0, w_ssim=0.0, w_grad=0.0, beta_kl=0.0):
        super().__init__()
        self.w = dict(mse=w_mse, l1=w_l1, ssim=w_ssim, grad=w_grad)
        self.beta_kl = beta_kl
        self.register_buffer("window", _gauss_window())

    def forward(self, xhat, x, mu=None, logvar=None, w=None):
        terms = {}
        if self.w["mse"]: terms["mse"] = _wmean((xhat - x) ** 2, w)
        if self.w["l1"]: terms["l1"] = _wmean((xhat - x).abs(), w)
        if self.w["ssim"]: terms["ssim"] = 1 - _wmean(ssim_map(xhat, x, self.window), w)
        if self.w["grad"]: terms["grad"] = gradient_loss(xhat, x, w)
        total = sum(self.w[k] * v for k, v in terms.items())
        if self.beta_kl and logvar is not None:
            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), 1))
            terms["kl"] = kl
            total = total + self.beta_kl * kl
        return total, {k: float(v.detach()) for k, v in terms.items()}


# === SECTION: training ===
def to_batch(u8, idx, augment=False, mask_u8=None):
    """uint8 GPU tensor [N,H,W] -> float [B,1,H,W] in [0,1] (+ validity map), optional random flips."""
    x = u8[idx].float().div_(255.0).unsqueeze(1)
    w = None if mask_u8 is None else (1.0 - mask_u8[idx].float()).unsqueeze(1)
    if augment:
        B = x.shape[0]
        fh = (torch.rand(B, device=x.device) < 0.5)[:, None, None, None]
        fv = (torch.rand(B, device=x.device) < 0.5)[:, None, None, None]
        flip = lambda t: torch.where(fv, torch.where(fh, t.flip(-1), t).flip(-2), torch.where(fh, t.flip(-1), t))
        x = flip(x)
        w = None if w is None else flip(w)
    return x, w


def train_autoencoder(model, images_u8, loss_fn, epochs=30, batch_size=64, lr=1e-3,
                      val_frac=0.1, augment=True, seed=42, log_every=1, sample_weights=None,
                      masks_u8=None):
    """Train on all crops (unlabelled) with a random 10% split held out for loss curves.

    masks_u8: optional no-data masks (1 = no-data); those pixels are excluded from the loss.
    sample_weights: optional per-image sampling weights.
    """
    seed_everything(seed)
    model = model.to(DEVICE)
    loss_fn = loss_fn.to(DEVICE)
    X = torch.from_numpy(images_u8).to(DEVICE)
    M = None if masks_u8 is None else torch.from_numpy(masks_u8).to(DEVICE)
    N = len(X)
    perm = np.random.permutation(N)
    n_val = int(N * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    steps = math.ceil(len(tr_idx) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.1)
    hist = []
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        if sample_weights is not None:
            p = sample_weights[tr_idx] / sample_weights[tr_idx].sum()
            order = np.random.choice(tr_idx, size=len(tr_idx), replace=True, p=p)
        else:
            order = np.random.permutation(tr_idx)
        run, parts_sum = 0.0, {}
        for s in range(steps):
            idx = torch.from_numpy(order[s * batch_size:(s + 1) * batch_size]).to(DEVICE)
            x, w = to_batch(X, idx, augment, M)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                xhat, mu, logvar = model(x)
            loss, parts = loss_fn(xhat.float(), x, mu.float(), None if logvar is None else logvar.float(), w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            run += loss.item()
            for k, v in parts.items(): parts_sum[k] = parts_sum.get(k, 0) + v
        val = evaluate_recon(model, X, val_idx, M)
        rec = dict(epoch=ep + 1, train_loss=run / steps, **{f"train_{k}": v / steps for k, v in parts_sum.items()},
                   val_mse=val["mse"], val_ssim=val["ssim"], sec=time.time() - t0)
        hist.append(rec)
        if (ep + 1) % log_every == 0:
            print(f"ep {ep+1:3d}/{epochs} loss {rec['train_loss']:.4f} | val MSE {val['mse']:.5f} "
                  f"SSIM {val['ssim']:.4f} | {rec['sec']:.1f}s")
    return model, pd.DataFrame(hist), val_idx


@torch.no_grad()
def _batch_metrics(model, X, idx, M, window):
    x, w = to_batch(X, idx, False, M)
    xhat, mu, _ = model(x)
    xhat = xhat.float()
    if w is None:
        w = torch.ones_like(x)
    den = w.sum(dim=(1, 2, 3)).clamp_min(1.0)
    mse = (((xhat - x) ** 2) * w).sum(dim=(1, 2, 3)) / den
    ssim = (ssim_map(xhat, x, window) * w).sum(dim=(1, 2, 3)) / den
    return mu.float(), mse, ssim


@torch.no_grad()
def evaluate_recon(model, X, idx, M=None, batch_size=128):
    """Mean per-image MSE / SSIM over idx, computed on valid (non-no-data) pixels if M is given."""
    model.eval()
    window = _gauss_window(device=DEVICE)
    mse, ssim = [], []
    for s in range(0, len(idx), batch_size):
        b = torch.as_tensor(idx[s:s + batch_size], device=DEVICE)
        _, m_, s_ = _batch_metrics(model, X, b, M, window)
        mse.append(m_); ssim.append(s_)
    return dict(mse=torch.cat(mse).mean().item(), ssim=torch.cat(ssim).mean().item())


@torch.no_grad()
def embed_and_reconstruct(model, images_u8, masks_u8=None, batch_size=128):
    """Latent vectors (mu) for every crop, plus per-image reconstruction MSE / SSIM (valid pixels)."""
    model.eval().to(DEVICE)
    X = torch.from_numpy(images_u8).to(DEVICE)
    M = None if masks_u8 is None else torch.from_numpy(masks_u8).to(DEVICE)
    window = _gauss_window(device=DEVICE)
    Z, mse, ssim = [], [], []
    for s in range(0, len(X), batch_size):
        idx = torch.arange(s, min(s + batch_size, len(X)), device=DEVICE)
        z, m_, s_ = _batch_metrics(model, X, idx, M, window)
        Z.append(z.cpu().numpy()); mse.append(m_.cpu().numpy()); ssim.append(s_.cpu().numpy())
    return np.concatenate(Z), np.concatenate(mse), np.concatenate(ssim)


@torch.no_grad()
def residual_images(model, images_u8, masks_u8=None, gain=4.0, batch_size=128):
    """Stage-2 input of the two-stage autoencoder: the signed reconstruction residual of a trained
    stage-1 model, amplified and mapped to uint8 as 128 + 127*clip(gain*(x - x_hat), -1, 1).
    No-data pixels are set to 128 (zero residual)."""
    model.eval().to(DEVICE)
    X = torch.from_numpy(images_u8).to(DEVICE)
    M = None if masks_u8 is None else torch.from_numpy(masks_u8).to(DEVICE)
    out = np.empty_like(images_u8)
    for s in range(0, len(X), batch_size):
        idx = torch.arange(s, min(s + batch_size, len(X)), device=DEVICE)
        x, w = to_batch(X, idx, False, M)
        r = x - model(x)[0].float()
        if w is not None:
            r = r * w
        out[s:s + len(idx)] = (128 + 127 * (gain * r).clamp(-1, 1)).round().squeeze(1).byte().cpu().numpy()
    return out


def load_model(v, out_dir):
    m = ConvAE(**CONFIGS[v]["model"])
    m.load_state_dict(torch.load(os.path.join(out_dir, v, "model.pt"), map_location=DEVICE))
    return m.to(DEVICE).eval()


def version_latents(v, X, out_dir, mask_cache=None):
    """Final fixed-length 1-D latent per crop for version v, plus the stage-1 reconstruction
    MSE / SSIM. Two-stage versions (residual_of=...) concatenate the stage-1 scene code with the
    stage-2 code of the stage-1 reconstruction residual."""
    cfg = CONFIGS[v]
    if cfg.get("residual_of"):
        base = cfg["residual_of"]
        Xin, M = preprocess(CONFIGS[base], X, mask_cache)       # stage 1 sees exactly its own input
        s1 = load_model(base, out_dir)
        Z1, mse, ssim = embed_and_reconstruct(s1, Xin, M)
        Z2, _, _ = embed_and_reconstruct(load_model(v, out_dir), residual_images(s1, Xin, M), M)
        return np.hstack([Z1, Z2]), mse, ssim
    Xin, M = preprocess(cfg, X, mask_cache)
    return embed_and_reconstruct(load_model(v, out_dir), Xin, M)


@torch.no_grad()
def reconstruct(model, images_u8, indices):
    model.eval().to(DEVICE)
    x = torch.from_numpy(images_u8[indices]).to(DEVICE).float().div(255).unsqueeze(1)
    return x.squeeze(1).cpu().numpy(), model(x)[0].float().squeeze(1).cpu().numpy()


# === SECTION: iforest ===
from sklearn.ensemble import IsolationForest


def novelty_scores(Z, n_estimators=500, max_samples=256, seeds=(0, 1, 2, 3, 4)):
    """Isolation Forest novelty score, averaged over seeds for stability.

    sklearn's score_samples() returns the *negated* anomaly score of Liu et al.
    (2008), i.e. higher = more normal. We negate it back, so the returned
    Novelty Score equals s(x, n) in (0, 1]: HIGHER = MORE ANOMALOUS, which is the
    competition convention. contamination='auto' only affects sklearn's
    predict() labels, which we never use: the anomaly count comes from the
    statistical threshold on these continuous scores.
    """
    runs = []
    for s in seeds:
        forest = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                 contamination="auto", random_state=s, n_jobs=-1).fit(Z)
        runs.append(-forest.score_samples(Z))
    runs = np.stack(runs)
    return runs.mean(0), runs.std(0)


# === SECTION: threshold ===
def mad_threshold(scores, k=3.5):
    """Iglewicz & Hoaglin (1993) modified z-score: M = 0.6745 (x - median) / MAD, flag M > k."""
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    return med + k * mad / 0.6745


def kneedle_threshold(scores):
    """Elbow of the ascending sorted-score curve: the point farthest from the chord joining
    the curve's endpoints after min-max normalisation (Satopaa et al. 2011, 'Kneedle')."""
    y = np.sort(scores)
    x = np.linspace(0, 1, len(y))
    yn = (y - y[0]) / (y[-1] - y[0])
    i = int(np.argmax(x - yn))       # convex increasing curve: max distance below the chord
    return y[i], i


def gmm_threshold(scores, seed=0):
    """Two-component 1-D Gaussian mixture on the scores; boundary where the posterior of the
    high-mean component first exceeds 0.5."""
    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(2, random_state=seed).fit(scores.reshape(-1, 1))
    hi = int(np.argmax(g.means_.ravel()))
    grid = np.linspace(scores.min(), scores.max(), 5000).reshape(-1, 1)
    post = g.predict_proba(grid)[:, hi]
    above = np.where((post > 0.5) & (grid.ravel() > g.means_.ravel()[1 - hi]))[0]
    return float(grid[above[0], 0]) if len(above) else float("nan"), g


def distribution_diagnostics(scores):
    k2, p = stats.normaltest(scores)
    return dict(n=len(scores), mean=scores.mean(), std=scores.std(), median=np.median(scores),
                mad=np.median(np.abs(scores - np.median(scores))),
                skew=stats.skew(scores), excess_kurtosis=stats.kurtosis(scores),
                dagostino_k2=k2, normality_p=p)


# === SECTION: heatmaps ===
def error_map(orig, recon, smooth_sigma=2.0, valid=None):
    """Per-pixel squared error, lightly Gaussian-smoothed so structure is readable.
    If a validity map is given, no-data pixels are zeroed (they carry no terrain information)."""
    e = (orig - recon) ** 2
    if valid is None:
        return ndimage.gaussian_filter(e, smooth_sigma) if smooth_sigma else e
    if not smooth_sigma:
        return e * valid
    # validity-weighted smoothing: no-data pixels neither contribute error nor leak it into valid ones
    num = ndimage.gaussian_filter(e * valid, smooth_sigma)
    den = ndimage.gaussian_filter(valid.astype(float), smooth_sigma)
    return np.where(valid > 0, num / np.maximum(den, 1e-6), 0.0)


def heatmap_descriptors(orig, err, valid=None):
    """Quantitative descriptors of an error map that support the physical hypotheses.

    concentration : share of total error in the top 5% of pixels (1.0 = all in a few pixels,
                    ~0.05 = perfectly diffuse)
    line_score    : strongest single-row or single-column mean error relative to the median
                    row/column (a straight seam gives a sharp spike)
    seam_step     : largest mean-intensity jump between the two halves on either side of any
                    horizontal/vertical line (spliced terrain with mismatched albedo)
    periodicity   : fraction of non-DC spectral power held by the 20 strongest frequencies
                    (stripes / banding from sensor read-out)
    saturation    : fraction of valid pixels at 0 or 1 (clipped / dead / hot pixels)
    """
    v = np.ones_like(orig, bool) if valid is None else valid.astype(bool)
    flat = np.sort(err[v].ravel())[::-1]
    conc = flat[: max(1, len(flat) // 20)].sum() / max(flat.sum(), 1e-12)
    rows = (err * v).sum(1) / np.maximum(v.sum(1), 1)          # row / column means over valid pixels only
    cols = (err * v).sum(0) / np.maximum(v.sum(0), 1)
    line = max(rows.max() / (np.median(rows) + 1e-12), cols.max() / (np.median(cols) + 1e-12))
    cr, cc = np.cumsum(orig.mean(1)), np.cumsum(orig.mean(0))
    n = orig.shape[0]
    steps = []
    for c in (cr, cc):
        for i in range(20, n - 20):
            steps.append(abs(c[i - 1] / i - (c[-1] - c[i - 1]) / (n - i)))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(orig - orig.mean()))) ** 2
    spec[n // 2, n // 2] = 0
    top = np.sort(spec.ravel())[::-1]
    o = orig[v]
    return dict(concentration=conc, line_score=line, seam_step=max(steps),
                periodicity=top[:20].sum() / top.sum(),
                saturation=float(((o <= 1 / 255) | (o >= 254 / 255)).mean()),
                mean=o.mean(), std=o.std())
