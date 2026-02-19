import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
from netneurotools import stats

# settings
savefig = True
N_SPIN = 10_000
SPIN_SEED = 12
gm_value_col = "t1_24h_mean"

# paths
root = Path("/Users/tahminehtaheri/aquaporin_four")
aqp4_fp = root / "data" / "AHBA_gene_expression" / "Cammoun" / "aqp4_cammoun_label.csv"
gm_fp = root / "data" / "MRI_tracer" / "ringstad_REF_table_cortical.csv"
cen_fp = root / "data" / "coordination" / "cammoun2012_MNI152NLin2009aSym_scale033_centroids_metadata.csv"
out_fp = root / "figures" / "MRI_tracer" / "scatter_AQP4_vs_tracer_cortex_pspin.svg"

# load
AQP4 = pd.read_csv(aqp4_fp)
GM = pd.read_csv(gm_fp)
cen = pd.read_csv(cen_fp)

# clean AQP4
AQP4 = AQP4.copy()
AQP4["label"] = AQP4["label"].astype(str).str.strip().str.lower()
AQP4["hemisphere"] = AQP4["hemisphere"].astype(str).str.strip().str.upper()
AQP4["structure"] = AQP4["structure"].astype(str).str.strip().str.lower()
AQP4[["x_mni", "y_mni", "z_mni"]] = cen[["x_mni", "y_mni", "z_mni"]].values


AQP4 = AQP4[AQP4["structure"].eq("cortex")].copy()

# parse GM cortex regions
gm_region_col = "region"
def gm_to_key(s):
    s = str(s).strip()
    if s.startswith("ctx-lh-"):
        return ("L", s.replace("ctx-lh-", "").lower())
    if s.startswith("ctx-rh-"):
        return ("R", s.replace("ctx-rh-", "").lower())
    return (None, None)

GM = GM.copy()
GM[["gm_hemi", "gm_label"]] = GM[gm_region_col].apply(lambda v: pd.Series(gm_to_key(v)))
GM = GM.dropna(subset=["gm_hemi", "gm_label"]).copy()

# merge
AQP4_key = AQP4.rename(columns={"label": "gm_label", "hemisphere": "gm_hemi"})[
    ["gm_label", "gm_hemi", "AQP4", "x_mni", "y_mni", "z_mni"]
]
df = GM.merge(AQP4_key, on=["gm_label", "gm_hemi"], how="inner")

# keep only valid rows
df = df[["gm_hemi", "gm_label", "AQP4", "x_mni", "y_mni", "z_mni", gm_value_col]].dropna().copy()

x = df["AQP4"].to_numpy(float)
y = df[gm_value_col].to_numpy(float)
coords = df[["x_mni", "y_mni", "z_mni"]].to_numpy(float)
hemiid = (df["gm_hemi"].to_numpy(str) == "L").astype(int)  # 1=left, 0=right

# correlation
r_obs = pearsonr(x, y)[0]

# spin nulls
spins = stats.gen_spinsamples(coords, hemiid=hemiid, n_rotate=N_SPIN, seed=SPIN_SEED)
r_null = np.array([pearsonr(x, y[spins[:, i]])[0] for i in range(N_SPIN)], float)

# p_spin
if r_obs >= 0:
    p_spin = (1 + np.sum(r_null >= r_obs)) / (1 + r_null.size)
else:
    p_spin = (1 + np.sum(r_null <= r_obs)) / (1 + r_null.size)

print(f"n={len(x)} | r={r_obs:.3f} | p_spin={p_spin:.3g}")

# plot
grayln = "#6b7280"
cortexcol = "tab:blue"

fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=300)

ax.scatter(
    x, y,
    s=40, alpha=0.75,
    edgecolors="black", linewidths=0.15,
    color=cortexcol
)

m, b = np.polyfit(x, y, 1)
xx = np.linspace(x.min(), x.max(), 200)
ax.plot(xx, m * xx + b, linewidth=1.6, color=grayln, alpha=0.9)

ax.text(
    0.06, 0.12,
    f"r = {r_obs:.2f}\n$p_\\mathrm{{spin}}$ = {p_spin:.2g}",
    transform=ax.transAxes, ha="left", va="top", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
)

ax.set_xlabel("AQP4 (Cammoun cortex)")
ax.set_ylabel(gm_value_col)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# save
if savefig:
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, format="svg", dpi=300, bbox_inches="tight")
    print("Saved:", out_fp)

plt.show()
plt.close(fig)
