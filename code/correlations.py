
##plot scatter plos of: AQP4 vs eight neurodegenerative diseases / edema / blood perfusion / vein desity
##plot correlations of AQP1/AQP4/AQP9 vs above phenotypes
##plot heatmap of correlation between AQP1, AQP4, and AQP9

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr
from brainspace.null_models.moran import MoranRandomization
from statsmodels.stats.multitest import multipletests


def load_xyz(fp: Path) -> np.ndarray:
    df = pd.read_csv(fp, header=None)
    return df.iloc[:, :3].to_numpy(float)


def invdist_w(xyz: np.ndarray) -> np.ndarray:
    d = distance_matrix(xyz, xyz)
    np.fill_diagonal(d, np.inf)
    return 1.0 / d


def moran_surrogates(x: np.ndarray, w: np.ndarray, n_surr: int, seed: int) -> np.ndarray:
    x = np.asarray(x, float).copy()
    if not np.isfinite(x).all():
        x[~np.isfinite(x)] = np.nanmean(x)
    mr = MoranRandomization(n_rep=n_surr, random_state=seed, tol=1e-8).fit(w)
    return mr.randomize(x)


def msr_pvalue_one_sided_signed(x_full, y_full, x_surr_full, mask):
    x = np.asarray(x_full, float)[mask]
    y = np.asarray(y_full, float)[mask]

    r_obs = pearsonr(x, y)[0]

    Xs = x_surr_full[:, mask]  # (n_surr, n_kept)
    Xs = (Xs - Xs.mean(axis=1, keepdims=True)) / (Xs.std(axis=1, keepdims=True) + 1e-12)
    y0 = (y - y.mean()) / (y.std() + 1e-12)

    r_null = (Xs * y0).mean(axis=1)

    if r_obs >= 0:
        p = (1 + np.sum(r_null >= r_obs)) / (1 + r_null.size)
    else:
        p = (1 + np.sum(r_null <= r_obs)) / (1 + r_null.size)

    return r_obs, p


def scatter_panel(x_full, y_full, sub_mask, out_fp, ylab, stat_text):
    mask = np.isfinite(x_full) & np.isfinite(y_full)
    x = np.asarray(x_full, float)[mask]
    y = np.asarray(y_full, float)[mask]
    sub = np.asarray(sub_mask, bool)[mask]

    from matplotlib import cm
    cw = cm.get_cmap("coolwarm")
    magenta = cw(0.92)
    cortexcol = "tab:blue"
    grayln = "#6b7280"

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=300)

    ax.scatter(x[sub], y[sub], s=30, alpha=0.7, label="Subcortex",
               edgecolors="black", linewidths=0.1, color=magenta)
    ax.scatter(x[~sub], y[~sub], s=30, alpha=0.7, label="Cortex",
               edgecolors="black", linewidths=0.1, color=cortexcol)

    m, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(xx, m * xx + b, linewidth=1.6, color=grayln, alpha=0.9)

    ax.text(
        0.06, 0.12, stat_text,
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )

    ax.set_xlabel("AQP4")
    ax.set_ylabel(ylab)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0, frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.12, right=0.78, bottom=0.12, top=0.97)
    fig.savefig(out_fp, format="svg", dpi=300)

    plt.show()
    plt.close(fig)


def triangle_heatmap(gene_dict, out_fp):
    genes = list(gene_dict.keys())
    G = len(genes)

    C = np.full((G, G), np.nan, float)
    for i, gi in enumerate(genes):
        xi = np.asarray(gene_dict[gi], float).ravel()
        for j, gj in enumerate(genes):
            xj = np.asarray(gene_dict[gj], float).ravel()
            m = np.isfinite(xi) & np.isfinite(xj)
            if m.sum() > 2:
                C[i, j] = pearsonr(xi[m], xj[m])[0]

    C_plot = C.copy()
    C_plot[np.triu_indices_from(C_plot, k=1)] = np.nan

    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=300)
    im = ax.imshow(C_plot, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(range(G))
    ax.set_yticks(range(G))
    ax.set_xticklabels(genes, rotation=45, ha="right")
    ax.set_yticklabels(genes)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, format="svg", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def lollipop_multi_gene(gene_dict, phenos, out_fp):
    genes = ["AQP1", "AQP4", "AQP9"]
    genes = [g for g in genes if g in gene_dict]

    colors = {"AQP1": "tab:red", "AQP4": "tab:blue", "AQP9": "tab:gray"}

    names = [n for n, _ in phenos]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(0.60 * len(names) + 3.0, 4.8), dpi=300)

    step = 0.18
    if len(genes) == 1:
        offsets = [0.0]
    elif len(genes) == 2:
        offsets = [-step, step]
    else:
        offsets = [-step, 0.0, step]

    for gi, g in enumerate(genes):
        vec = np.asarray(gene_dict[g], float).ravel()

        rvals = []
        for _, v in phenos:
            v = np.asarray(v, float).ravel()
            m = np.isfinite(vec) & np.isfinite(v)
            rvals.append(pearsonr(vec[m], v[m])[0])

        rvals = np.asarray(rvals, float)
        xi = x + offsets[gi]

        for xj, rj in zip(xi, rvals):
            ax.vlines(xj, 0, rj, linewidth=1.2, color=colors[g])

        ax.scatter(xi, rvals, s=22, color=colors[g], label=g, zorder=3)

    ax.axhline(0, linewidth=1.0, color="lightgray")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Pearson r")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    out_fp.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_fp, format="svg", dpi=300)
    plt.show()
    plt.close(fig)

N = 400
N_SUBCORTEX = 54
N_SURR = 10_000
SEED = 12

# paths
root = Path("/Users/tahminehtaheri/aquaporin_four")

gene_dir = root / "data" / "AHBA_gene_expression" / "Schaefer400_MelbourneS4"
aqp1_fp = gene_dir / "aqp1_values.npy"
aqp4_fp = gene_dir / "aqp4_values.npy"
aqp9_fp = gene_dir / "aqp9_values.npy"

coords_fp = root / "data" / "coordination" / f"COG{N}_label.csv"

disease_csv = (root / "data" / "neurodegenerative_disease" / "Schaefer400_MelbourneS4"
               / f"atrophy_VBM_Tstat_parcellated_all_{N}.csv")
disease_order = ["EOA", "PS1", "3Rtau", "4Rtau", "TDP43A", "TDP43C", "DLB", "LOA"]

edema_fp = (root / "data" / "edema" / "Schaefer400_MelbourneS4"
            / f"edema_1094_400.npy")

vascular_dir = root / "data" / "vascular_measures" / "Schaefer400_MelbourneS4"
cbf_fp = vascular_dir / f"perfusion_{N}.npy"
vden_fp = vascular_dir / f"vdensity_{N}.npy"

fig_scatter = root / "figures" / "scatterplots"
fig_misc = root / "figures" / "aqps"
res_dir = root / "results"


# load vectors 
aqp1 = np.load(aqp1_fp).astype(float).ravel()
aqp4 = np.load(aqp4_fp).astype(float).ravel()
aqp9 = np.load(aqp9_fp).astype(float).ravel()

xyz = load_xyz(coords_fp)
w = invdist_w(xyz)

sub_mask = np.zeros(aqp4.size, dtype=bool)
sub_mask[:N_SUBCORTEX] = True

disease_df = pd.read_csv(disease_csv)
Y_dis = disease_df[disease_order].to_numpy(float)

edema = np.load(edema_fp).astype(float).ravel()

cbf = np.load(cbf_fp).astype(float).ravel()
vden = np.load(vden_fp).astype(float).ravel()

# flip t-statistics of diseases so “more atrophy” is bigger
Y_dis = -1.0 * Y_dis


# Moran surrogates (load or compute) 
cache_fp = res_dir / f"aqp4_moran_{N_SURR}.npy"
if cache_fp.exists():
    aqp4_surr = np.load(cache_fp)
else:
    aqp4_surr = moran_surrogates(aqp4, w, n_surr=N_SURR, seed=SEED)
    cache_fp.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_fp, aqp4_surr)


# MSR p for diseases (then FDR across 8)
pvals = []
rvals = []
masks = []

for j, name in enumerate(disease_order):
    y = Y_dis[:, j]
    m = np.isfinite(aqp4) & np.isfinite(y)
    r, p = msr_pvalue_one_sided_signed(aqp4, y, aqp4_surr, m)
    rvals.append(r)
    pvals.append(p)
    masks.append(m)

pvals = np.array(pvals, float)
rvals = np.array(rvals, float)

_, qvals, _, _ = multipletests(pvals, method="fdr_bh")


# disease scatterplots
for j, name in enumerate(disease_order):
    y = Y_dis[:, j]
    txt = f"r = {rvals[j]:.4f}\n$q_\\mathrm{{MSR}}$ = {qvals[j]:.4g}"
    out_fp = fig_scatter / f"scatter_AQP4_vs_{name}.svg"
    scatter_panel(aqp4, y, sub_mask, out_fp, ylab=name, stat_text=txt)


# edema scatterplot 
m_ed = np.isfinite(aqp4) & np.isfinite(edema)
r_ed, p_ed = msr_pvalue_one_sided_signed(aqp4, edema, aqp4_surr, m_ed)

txt_ed = f"r = {r_ed:.4f}\n$p_\\mathrm{{MSR}}$ = {p_ed:.4g}"
out_ed = fig_scatter / "scatter_AQP4_vs_Edema.svg"
scatter_panel(aqp4, edema, sub_mask, out_ed, ylab="Edema", stat_text=txt_ed)

# AQP4 vs CBF scatterplot
m_cbf = np.isfinite(aqp4) & np.isfinite(cbf)
r_cbf, p_cbf = msr_pvalue_one_sided_signed(aqp4, cbf, aqp4_surr, m_cbf)
txt_cbf = f"r = {r_cbf:.4f}\n$p_\\mathrm{{MSR}}$ = {p_cbf:.4g}"
out_cbf = fig_scatter / "scatter_AQP4_vs_CBF.svg"
scatter_panel(aqp4, cbf, sub_mask, out_cbf, ylab="CBF", stat_text=txt_cbf)

# AQP4 vs VeinDensity scatterplot
m_vd = np.isfinite(aqp4) & np.isfinite(vden)
r_vd, p_vd = msr_pvalue_one_sided_signed(aqp4, vden, aqp4_surr, m_vd)
txt_vd = f"r = {r_vd:.4f}\n$p_\\mathrm{{MSR}}$ = {p_vd:.4g}"
out_vd = fig_scatter / "scatter_AQP4_vs_VeinDensity.svg"
scatter_panel(aqp4, vden, sub_mask, out_vd, ylab="Vein density", stat_text=txt_vd)


# AQP1/AQP4/AQP9 correlation heatmap 
gene_dict = {"AQP1": aqp1, "AQP4": aqp4, "AQP9": aqp9}
triangle_heatmap(gene_dict, fig_misc / "aqp1_aqp4_aqp9_heatmap.svg")


# lollipop: AQP4 vs vascular + diseases + edema 
phenos = [("CBF", cbf), ("VeinDensity", vden)]
phenos += [(nm, Y_dis[:, j]) for j, nm in enumerate(disease_order)]
phenos += [("Edema", edema)]

gene_dict = {"AQP1": aqp1, "AQP4": aqp4, "AQP9": aqp9}
lollipop_multi_gene(gene_dict, phenos, fig_misc / "aqp1_aqp4_aqp9_vs_maps_lollipop.svg")

