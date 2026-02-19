## scatterplot AHBA RNA-seq vs microarray (with Moran p_MSR)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import pearsonr
from scipy.spatial import distance_matrix
from brainspace.null_models.moran import MoranRandomization

# settings
savefig = True
N_SUBCORTEX = 54
N_SURR = 10_000
SEED = 12

# paths
root = Path("/Users/tahminehtaheri/aquaporin_four")

micro_fp  = root / "data" / "AHBA_gene_expression" / "Schaefer400_MelbourneS4" / "aqp4_values.npy"
rnaseq_fp = root / "data" / "RNA_seq" / "AHBA" / "abagen_rnaseq_interpolation_normalized.csv"
coords_fp = root / "data" / "coordination" / "COG400_label.csv"   # 4 cols: x,y,z,label (no header)

out_fp = root / "figures" / "RNA_seq" / "scatter_AQP4_micro_vs_rnaseq.svg"
cache_fp = root / "results" / f"aqp4_micro_moran_{N_SURR}.npy"

# load data
aqp4_micro = np.load(micro_fp).astype(float).ravel()
rna_seq = pd.read_csv(rnaseq_fp)
aqp4_seq = rna_seq["AQP4"].to_numpy(dtype=float).ravel()

# load coords (x,y,z are first 3 columns; no header)
coords_df = pd.read_csv(coords_fp, header=None)
xyz = coords_df.iloc[:, :3].to_numpy(float)

# match length across arrays
n = min(aqp4_micro.size, aqp4_seq.size, xyz.shape[0])
aqp4_micro = aqp4_micro[:n]
aqp4_seq   = aqp4_seq[:n]
xyz        = xyz[:n, :]

# correlation
mask = np.isfinite(aqp4_micro) & np.isfinite(aqp4_seq)
x = aqp4_micro[mask]
y = aqp4_seq[mask]
r_obs, p_pear = pearsonr(x, y)
print(f"Pearson AQP4 (microarray vs RNA-seq): r = {r_obs:.3f}, p = {p_pear:.2e}")

# Moran surrogates for x (microarray)
coords_use = xyz[mask]
D = distance_matrix(coords_use, coords_use)
np.fill_diagonal(D, np.inf)
W = 1.0 / D

# cache surrogates (optional)
if cache_fp.exists():
    x_surr = np.load(cache_fp)
    # if cached array doesn't match current n_kept, recompute
    if x_surr.shape[1] != x.size or x_surr.shape[0] != N_SURR:
        x_surr = None
else:
    x_surr = None

if x_surr is None:
    mr = MoranRandomization(n_rep=N_SURR, random_state=SEED, tol=1e-8).fit(W)
    x_surr = mr.randomize(x)  # (N_SURR, n_kept)
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_fp, x_surr)

r_null = np.array([pearsonr(x_surr[i], y)[0] for i in range(N_SURR)], float)

if r_obs >= 0:
    p_msr = (1 + np.sum(r_null >= r_obs)) / (1 + r_null.size)
else:
    p_msr = (1 + np.sum(r_null <= r_obs)) / (1 + r_null.size)

print(f"p_MSR(one-sided, signed) = {p_msr:.3g}")

# plot
sub = np.arange(x.size) < N_SUBCORTEX

cw = cm.get_cmap("coolwarm")
magenta = cw(0.92)
cortexcol = "tab:blue"
grayln = "#6b7280"

fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)

ax.scatter(x[sub], y[sub], s=40, alpha=0.7, edgecolors="black",
           linewidths=0.1, color=magenta, label="Subcortex")
ax.scatter(x[~sub], y[~sub], s=40, alpha=0.7, edgecolors="black",
           linewidths=0.1, color=cortexcol, label="Cortex")

m, b = np.polyfit(x, y, 1)
xx = np.linspace(x.min(), x.max(), 200)
ax.plot(xx, m * xx + b, linewidth=1.6, color=grayln, alpha=0.9)

ax.text(
    0.06, 0.12,
    f"r = {r_obs:.2f}\n$p_\\mathrm{{MSR}}$ = {p_msr:.2g}",
    transform=ax.transAxes, ha="left", va="top", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
)

ax.set_xlabel("AQP4 microarray")
ax.set_ylabel("AQP4 RNA-seq")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(rect=[0, 0, 0.88, 1])

# save
if savefig:
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, format="svg", dpi=300, bbox_inches="tight")
    print("Saved:", out_fp)

plt.show()
plt.close(fig)
