"""PCA plot, nearest 1000G populations and supervised ADMIXTURE proportions for the sample."""

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import config_arg, load_config  # noqa: E402

POP_NAMES = {
    "ACB": "African Caribbean (Barbados)", "ASW": "African American (SW USA)", "ESN": "Esan (Nigeria)",
    "GWD": "Gambian", "LWK": "Luhya (Kenya)", "MSL": "Mende (Sierra Leone)", "YRI": "Yoruba (Nigeria)",
    "CLM": "Colombian (Medellín)", "MXL": "Mexican (Los Angeles)", "PEL": "Peruvian (Lima)", "PUR": "Puerto Rican",
    "CDX": "Dai Chinese", "CHB": "Han Chinese (Beijing)", "CHS": "Southern Han Chinese", "JPT": "Japanese",
    "KHV": "Kinh (Vietnam)", "CEU": "NW European (Utah)", "FIN": "Finnish", "GBR": "British",
    "IBS": "Iberian (Spain)", "TSI": "Tuscan (Italy)", "BEB": "Bengali", "GIH": "Gujarati",
    "ITU": "Telugu", "PJL": "Punjabi", "STU": "Sri Lankan Tamil",
}
# Fixed categorical order (validated reference palette, slots 1-5)
SUPERPOP_COLORS = {"AFR": "#2a78d6", "AMR": "#eb6834", "EAS": "#1baf7a", "EUR": "#eda100", "SAS": "#e87ba4"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"


def load_pcs(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    out = cfg["out_dir"]
    pcs = pd.read_csv(out / "ancestry.eigenvec", sep="\t").rename(columns=lambda c: c.lstrip("#"))
    panel = pd.read_csv(cfg["ref_dir"] / "1kg" / "integrated_call_samples_v3.20130502.ALL.panel",
                        sep="\t", usecols=["sample", "pop", "super_pop"])
    pc_cols = [c for c in pcs.columns if c.startswith("PC")]
    return pcs.merge(panel, left_on="IID", right_on="sample", how="left"), pc_cols


def nearest_populations(pcs: pd.DataFrame, pc_cols: list[str], me: pd.Series, eigval: np.ndarray) -> pd.DataFrame:
    """Distance from the sample to each population centroid, PCs weighted by sqrt(eigenvalue)."""
    w = np.sqrt(eigval[: len(pc_cols)])
    centroids = pcs.dropna(subset=["pop"]).groupby(["pop", "super_pop"])[pc_cols].mean()
    dist = np.sqrt((((centroids - me[pc_cols].astype(float)) * w) ** 2).sum(axis=1))
    res = dist.rename("distance").reset_index().sort_values("distance")
    res["name"] = res["pop"].map(POP_NAMES)
    return res


def admixture(cfg: dict) -> dict[str, float]:
    out = cfg["out_dir"]
    q_file = next(out.glob("ancestry.*.Q"))
    q = pd.read_csv(q_file, sep=r"\s+", header=None)
    fam = pd.read_csv(out / "ancestry.fam", sep=r"\s+", header=None, usecols=[1], names=["IID"])
    pop = pd.read_csv(out / "ancestry.pop", header=None, names=["label"])
    # Map each Q column to the reference label whose samples load on it most
    q.columns = q[pop["label"] != "-"].groupby(pop["label"]).mean().idxmax().to_numpy()
    me = q[fam["IID"] == cfg["sample_id"]].iloc[0]
    return me.sort_values(ascending=False).round(4).to_dict()


def plot(pcs: pd.DataFrame, me: pd.Series, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=SURFACE)
    for ax, (x, y) in zip(axes, [("PC1", "PC2"), ("PC3", "PC4")]):
        ax.set_facecolor(SURFACE)
        for sp, color in SUPERPOP_COLORS.items():
            d = pcs[pcs["super_pop"] == sp]
            ax.scatter(d[x], d[y], s=10, c=color, edgecolors=SURFACE, linewidths=0.4, label=sp)
        ax.scatter(me[x], me[y], s=260, marker="*", c=INK, edgecolors=SURFACE, linewidths=2, label="you", zorder=5)
        ax.annotate("you", (me[x], me[y]), xytext=(8, 8), textcoords="offset points", color=INK, fontsize=10)
        ax.set_xlabel(x, color=MUTED)
        ax.set_ylabel(y, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="best")
    fig.suptitle("Position among 1000 Genomes samples", color=INK, x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    args = config_arg(__doc__).parse_args()
    cfg = load_config(args.config)
    out = cfg["out_dir"]

    pcs, pc_cols = load_pcs(cfg)
    me = pcs[pcs["IID"] == cfg["sample_id"]].iloc[0]
    eigval = np.loadtxt(out / "ancestry.eigenval")

    nearest = nearest_populations(pcs, pc_cols, me, eigval)
    print("Nearest 1000G populations (PC1-10):")
    print(nearest.head(8).to_string(index=False))

    props = admixture(cfg)
    print("\nSupervised ADMIXTURE:", props)

    plot(pcs, me, out / "ancestry_pca.png")
    with open(out / "ancestry.json", "w") as f:
        json.dump({"nearest_populations": nearest.head(8).to_dict(orient="records"), "admixture": props}, f, indent=2)
    print(f"\nPlot: {out / 'ancestry_pca.png'}")


if __name__ == "__main__":
    main()
