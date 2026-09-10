"""Render report/NSSC_DA_Report.pdf from results.json, the notebook figures, the notebook
narrative and CHANGELOG.md (HTML -> PDF via headless Edge)."""
import os, json, html, subprocess
import markdown
from notebook_text import T

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT, FIG, REP = [os.path.join(ROOT, p) for p in ("outputs", "outputs/figures", "report")]
os.makedirs(REP, exist_ok=True)
R = json.load(open(os.path.join(OUT, "results.json")))
md = lambda s: markdown.markdown(s, extensions=["tables", "fenced_code"])
fig = lambda name, cap: (f'<figure><img src="../outputs/figures/{name}"><figcaption>{cap}</figcaption></figure>'
                         if os.path.exists(os.path.join(FIG, name)) else "")


def table(rows, cols=None, fmt=4, hl=None):
    if not rows: return ""
    cols = cols or list(rows[0].keys())
    f = lambda v: f"{v:.{fmt}f}" if isinstance(v, float) else html.escape(str(v))
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr%s>%s</tr>" % (' class="hl"' if hl and hl(r) else "", "".join(f"<td>{f(r.get(c, ''))}</td>" for c in cols)) for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


top5 = R["top5"]
L, Mc = R["final_config"]["loss"], R["final_config"]["model"]
loss_str = " + ".join(f"{w:g}·{name}" for name, w in [("(1−SSIM)", L.get("w_ssim", 0)), ("L1", L.get("w_l1", 0)),
                                                        ("MSE", L.get("w_mse", 0)), ("Gradient-L1", L.get("w_grad", 0))] if w)
bottleneck = f"spatial 8×8×{Mc['spatial_latent']} bottleneck" if Mc.get("spatial_latent") else "linear bottleneck"
final_box = f"""
<div class="final">
<h3>Final answers</h3>
<ul>
<li><b>Model:</b> {R['final_version']} - convolutional autoencoder trained from scratch; {bottleneck} → fixed-length
{Mc['latent_dim']}-d 1-D latent; loss = {loss_str}{'; map-projection no-data wedges masked' if R['final_config'].get('nodata') else ''}</li>
<li><b>Novelty Score:</b> Isolation Forest s(x,n) on latents (higher = more anomalous), 500 trees × 5 seeds</li>
<li><b>Statistical threshold:</b> modified z-score &gt; 3.5 (median + 3.5·MAD/0.6745) = <b>{R['threshold']:.4f}</b></li>
<li><b>Crops flagged:</b> <b>{R['n_flagged']}</b> of {R['n_crops']} ({100 * R['n_flagged'] / R['n_crops']:.2f} %)</li>
<li><b>Top-5 analysed in Phase 3:</b> {', '.join(r['filename'] + f" ({r['novelty']:.3f})" for r in top5)}</li>
</ul></div>"""

sec = []
key_findings = md(T["final"].split("**Key findings**", 1)[1]) if "**Key findings**" in T["final"] else ""
sec.append(f"""<div class="cover"><h1>Unsupervised Anomaly Detection in Mars HiRISE Orbital Imagery</h1>
<p class="sub">NSSC 2026 · Data Analytics · IIT Kharagpur</p><p class="sub">{html.escape(T['title'].split('**Team:**')[1].split('·')[0].strip())}</p>{final_box}
<h3>Key findings</h3>{key_findings}</div>""")
sec.append("<h1>0. Data and sanity pass</h1>" + md(T["sanity"]) + fig("fig01_random_batch.png", "Fig 0.1 - Random batch") +
           fig("fig02_pixel_stats.png", "Fig 0.2 - Per-crop intensity, contrast and no-data statistics") +
           "<h2>0.2 No-data (map-projection) masking - introduced in v3</h2>" + md(T["nodata"]) +
           fig("fig02b_nodata_masks.png", "Fig 0.3 - No-data masks: projection wedges are masked; thin interior dark features are deliberately kept"))
sweep = ""
if R.get("sweep"):
    sweep = md(T["latent_sweep"]) + table(R["sweep"], ["latent_dim", "epochs", "val_mse", "val_ssim", "pca90", "tail_ratio", "n_flag_mad35"]) + \
            fig("fig03_latent_sweep.png", "Fig 1.1 - Latent dimensionality sweep") + md(T.get("latent_choice", ""))
sec.append("<h1>Phase 1 - Deep Latent Compression</h1><h2>1.1 Architecture design</h2>" + md(T["arch"]) + sweep +
           "<h2>1.2 Loss function design</h2>" + md(T["loss"]) +
           fig("fig04_training_curves.png", "Fig 1.2a - Training curves of v1-v4") +
           fig("fig05_recon_versions.png", "Fig 1.2b - Reconstructions across versions") +
           "<h2>1.3 Latent space visualisation</h2>" + md(T["tsne_intro"]) +
           fig("fig06_tsne_properties.png", "Fig 1.3a - t-SNE of final latents coloured by image properties") +
           fig("fig07_tsne_thumbs.png", "Fig 1.3b - t-SNE with nearest-crop thumbnails") + md(T["tsne_read"]) +
           fig("fig09_tsne_flagged.png", "Fig 1.3c - Flagged crops on the t-SNE map"))
d = R["diag"]
sec.append("<h1>Phase 2 - Isolation Forest Novelty Engine</h1><h2>2.1 Novelty scoring</h2>" + md(T["if_intro"]) +
           "<h2>2.2 Statistical thresholding</h2>" + md(T["thr_intro"]) +
           table([{k: d[k] for k in ("mean", "std", "median", "mad", "skew", "excess_kurtosis", "normality_p")}]) +
           table(R["candidates"], hl=lambda r: abs(r["threshold"] - R["threshold"]) < 1e-12) +
           fig("fig08_threshold_candidates.png", "Fig 2.2 - Sorted scores with candidate thresholds, histogram, Q-Q plot") +
           md(T["thr_choice"]) + f'<p class="answer">Final threshold = {R["threshold"]:.4f} → {R["n_flagged"]} crops flagged</p>' +
           "<h3>Contamination parameter</h3>" + md(T["contamination"]) +
           "<h2>2.3 Image location analysis</h2>" + md(T["loc_intro"]) + fig("fig10_location.png", "Fig 2.3 - Location, illumination and season") +
           table([{"test": k, "value": (f"rho={v[0]:.3f}, p={v[1]:.2g}" if isinstance(v, list) else f"{v:.4g}")} for k, v in R["location"].items()]) +
           table(R["per_src_top"], ["source_image_id", "n", "n_flag", "rate", "latitude", "longitude", "sun_angle", "season", "resolution", "p_bonf"]) +
           md(T["loc_read"]) + "<h2>2.4 Metadata fusion (optional)</h2>" + md(T["fusion_intro"]) +
           table([{k: v for k, v in R["fusion"].items() if not isinstance(v, list)}]) + md(T["fusion_read"]))
pct = {r["filename"]: r for r in R["top5_pct"]}
sec.append("<h1>Phase 3 - Reconstruction Interpretability</h1><h2>3.1 Pixel-wise error heatmaps</h2>" + md(T["hm_intro"]) +
           fig("fig11_heatmaps_top5.png", "Fig 3.1 - Top-5 thresholded crops: original, reconstruction, error, overlay") +
           table(top5, ["rank", "filename", "source", "novelty", "recon_mse", "concentration", "line_score", "seam_step", "periodicity", "saturation"]) +
           "<p>Percentile of each descriptor against 500 random normal crops:</p>" + table(R["top5_pct"], fmt=0) +
           fig("fig11b_supplementary_heatmaps.png", "Fig 3.1b - Decoder residual on visually noticed crops (flagged content anomalies vs missed local glitches)") +
           "<h2>3.2 Geological report</h2>" + md(T["geo_report"]))
sec.append("<h1>Phase 4 - Architecture Iteration & Design Journal</h1>" +
           table(R["versions"]) + fig("fig12_versions_hist.png", "Fig 4.1 - Novelty Score distributions per version (orange = crops with no-data border)") +
           fig("fig13_versions_top8.png", "Fig 4.2 - Highest-scoring crops per version") +
           ("<p>Rank of visually noticed spot-check crops in each version (diagnostic only, never used to set the threshold):</p>"
            + table(R["spot"], fmt=0) if R.get("spot") else "") +
           md(open(os.path.join(ROOT, "CHANGELOG.md"), encoding="utf-8").read().split("\n", 1)[1]))
sec.append("<h1>Appendix - all flagged crops</h1>" + table(R["flagged"], ["filename", "source_image_id", "novelty", "recon_mse", "latitude", "longitude", "sun_angle", "season", "resolution"]))

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; color: #1b1b1b; line-height: 1.45; }
h1 { font-size: 17pt; color: #8a1c1c; border-bottom: 2px solid #8a1c1c; padding-bottom: 3px; margin-top: 26px; page-break-before: always; }
.cover h1 { page-break-before: avoid; border: none; font-size: 22pt; margin-top: 60px; }
h2 { font-size: 13pt; color: #333; margin-top: 18px; } h3 { font-size: 11pt; }
.sub { color: #555; font-size: 12pt; margin: 2px 0; }
figure { margin: 10px 0; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 235mm; }
figcaption { font-size: 8.5pt; color: #555; margin-top: 3px; }
table { border-collapse: collapse; margin: 8px 0; font-size: 8pt; page-break-inside: auto; }
th, td { border: 1px solid #ccc; padding: 2px 5px; text-align: left; } th { background: #f1e4e4; }
tr.hl td { background: #fff3b0; font-weight: 600; }
.final { background: #fff6d6; border: 2px solid #d9a400; border-radius: 6px; padding: 8px 14px; margin-top: 24px; }
.answer { background: #fff3b0; border-left: 5px solid #d9a400; padding: 6px 10px; font-weight: 700; }
code { font-size: 8.5pt; background: #f4f4f4; padding: 0 2px; }
"""
page = f"<!doctype html><html><head><meta charset='utf-8'><title>NSSC DA Report</title><style>{CSS}</style></head><body>{''.join(sec)}</body></html>"
html_path = os.path.join(REP, "NSSC_DA_Report.html")
open(html_path, "w", encoding="utf-8").write(page)
pdf_path = os.path.join(REP, "NSSC_DA_Report.pdf")
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}",
                "file:///" + html_path.replace("\\", "/")], check=True, timeout=180)
print("wrote", pdf_path, os.path.getsize(pdf_path) // 1024, "KB")
