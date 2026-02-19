##scatterplot RNA_seq data from HPA vs AHBA microarray

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import pearsonr


def to_1d(x):
    x = np.asarray(x)
    if x.ndim == 2 and 1 in x.shape:
        return x.ravel()
    return x


def load_hpa_ntpm_table(hpa_tsv, gene_name):
    hpa = pd.read_csv(hpa_tsv, sep="\t")
    hpa = hpa.rename(columns={"Gene name": "gene", "Subregion": "region", "nTPM": "ntpm"})
    hpa = hpa[["gene", "region", "ntpm"]]
    hpa = hpa[~hpa["region"].str.contains("white matter", case=False, na=False)]
    hpa = hpa[hpa["gene"] == gene_name]
    hpa = pd.pivot(hpa, columns="gene", index="region", values="ntpm")
    return np.log10(hpa + 1)

savefig = True
gene = "AQP4"

# paths
root = Path("/Users/tahminehtaheri/aquaporin_four")

hpa_dir = root / "data" / "RNA_seq" / "HPA"
ahba_dir = root / "data" / "AHBA_gene_expression" / "Destrieux_MelbourneS1"

hpa_tsv = hpa_dir / "hpa_whole-brain.tsv"
destrieux_labels_csv = hpa_dir / "destrieux_labels.csv"
hpa_destrieux_map_csv = hpa_dir / "hpa_destrieux_map.csv"
hpa_tian_map_csv = hpa_dir / "hpa_tians1_map.csv"
tian_label_txt = hpa_dir / "Tian_Subcortex_S1_3T_label.txt"

ahba_destrieux_csv = ahba_dir / "abagen_genes_Destrieux.csv"
ahba_tian_csv = ahba_dir / "abagen_gene_expression_Tian_Subcortex_S1.csv"

out_fig = root / "figures" / "RNA_seq" / f"scatter_{gene}_AHBA_vs_HPA_ctx_plus_sub.svg"


# load HPA 
hpa = load_hpa_ntpm_table(hpa_tsv, gene)

# cortex: HPA -> Destrieux(R) -> AHBA
destrieux_labels = pd.read_csv(destrieux_labels_csv)
destrieux_labels = destrieux_labels[destrieux_labels["hemisphere"] == "R"]

hpa_map_ctx = pd.read_csv(hpa_destrieux_map_csv)
destrieux_id_map = destrieux_labels.set_index("label")["id"].to_dict()
hpa_map_ctx["destrieux_id"] = hpa_map_ctx["Destrieux region"].map(destrieux_id_map)
hpa_map_ctx = hpa_map_ctx.dropna(subset=["destrieux_id"]).copy()
hpa_map_ctx["destrieux_id"] = hpa_map_ctx["destrieux_id"].astype(int)

ahba_destrieux = pd.read_csv(ahba_destrieux_csv, index_col=0)

hpa_ctx_matched = hpa.reindex(hpa_map_ctx["HPA region"].values)
ahba_ctx_matched = ahba_destrieux.reindex(hpa_map_ctx["destrieux_id"].values)

hpa_ctx = to_1d(hpa_ctx_matched[gene].to_numpy(float))
ahba_ctx = to_1d(ahba_ctx_matched[gene].to_numpy(float))

# subcortex: HPA -> Tian S1 -> AHBA 
gene_tian = pd.read_csv(ahba_tian_csv, index_col=0)
ahba_sub_full = gene_tian[gene].to_numpy(float)

tian_labels = pd.read_csv(tian_label_txt, header=None)[0].astype(str).tolist()
tian_name_to_index = {name: i + 1 for i, name in enumerate(tian_labels)}  # 1-based

tian_map = pd.read_csv(hpa_tian_map_csv)
tian_map["tian_index"] = tian_map["Tian region"].map(tian_name_to_index)
tian_map = tian_map.dropna(subset=["tian_index"]).copy()
tian_map["tian_index"] = tian_map["tian_index"].astype(int)

hpa_sub_matched = hpa.reindex(tian_map["HPA region"].values)
tian_map["hpa_val"] = to_1d(hpa_sub_matched[gene].to_numpy(float))

sub_grouped = (
    tian_map.groupby(["tian_index", "Tian region"], as_index=False)["hpa_val"]
    .mean()
    .sort_values("tian_index")
)

hpa_sub = sub_grouped["hpa_val"].to_numpy(float)
ahba_sub = np.array([ahba_sub_full[i - 1] for i in sub_grouped["tian_index"].values], dtype=float)

# concat + stats
ahba_all = np.concatenate([ahba_ctx, ahba_sub])
hpa_all = np.concatenate([hpa_ctx, hpa_sub])

is_ctx = np.concatenate([np.ones(len(ahba_ctx), dtype=bool), np.zeros(len(ahba_sub), dtype=bool)])

mask = np.isfinite(ahba_all) & np.isfinite(hpa_all)
x_plot = ahba_all[mask]
y_plot = hpa_all[mask]
is_ctx_plot = is_ctx[mask]
is_sub_plot = ~is_ctx_plot

r, p = pearsonr(x_plot, y_plot)
print(f"Pearson {gene} (AHBA vs HPA): r = {r:.3f}, p = {p:.2e}")
print("Cortex points:", len(ahba_ctx))
print("Subcortex points:", len(ahba_sub))
print("Total points (before NaN mask):", len(ahba_all))
print("Total points used (after NaN mask):", x_plot.size)

# plot
cw = cm.get_cmap("coolwarm")
magenta = cw(0.92)
cortexcol = "tab:blue"
grayln = "#6b7280"

fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)

ax.scatter(
    x_plot[is_sub_plot], y_plot[is_sub_plot],
    s=40, alpha=0.7, edgecolors="black", linewidths=0.1,
    color=magenta, label="Subcortex"
)
ax.scatter(
    x_plot[is_ctx_plot], y_plot[is_ctx_plot],
    s=40, alpha=0.7, edgecolors="black", linewidths=0.1,
    color=cortexcol, label="Cortex"
)

m, b = np.polyfit(x_plot, y_plot, 1)
xx = np.linspace(x_plot.min(), x_plot.max(), 200)
ax.plot(xx, m * xx + b, linewidth=1.6, color=grayln, alpha=0.9)

ax.set_xlabel(f"AHBA {gene}")
ax.set_ylabel(f"HPA {gene} (log10 nTPM + 1)")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(rect=[0, 0, 0.88, 1])

if savefig:
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, format="svg", dpi=300, bbox_inches="tight")
    print("Saved:", out_fig)

plt.show()
plt.close(fig)
