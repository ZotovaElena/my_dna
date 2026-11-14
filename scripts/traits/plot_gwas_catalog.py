"""Plot distinct traits per rsID and the most significant GWAS Catalog results.

Bars are shaded by how many copies of the catalog's effect allele you carry.
"""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import config_arg, load_config

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
NEUTRAL_BAR = "#b5b3ab"
THRESHOLD = "#d45b45"
# Ordinal blue ramp (steps 250/400/600): more copies of the effect allele = darker.
COPY_COLORS = {0: "#86b6ef", 1: "#3987e5", 2: "#184f95"}
COPY_LABELS = {0: "0 copies", 1: "1 copy", 2: "2 copies"}
UNKNOWN_LABEL = "Not genotyped / allele unknown"
GENOME_WIDE = -np.log10(5e-8)
REQUIRED = {"rsid", "genotype", "effect_allele", "effect_allele_copies",
            "strand_ambiguous", "efo_trait", "reported_trait", "neg_log10_p"}


def trait_label(report: pd.DataFrame) -> pd.Series:
    """Standardised EFO trait, falling back to the free-text reported trait."""
    efo = report["efo_trait"].fillna("").astype(str).str.strip()
    reported = report["reported_trait"].fillna("").astype(str).str.strip()
    return efo.where(efo != "", reported).replace("", "Unlabeled trait")


def top_associations(report: pd.DataFrame, top: int, per_rsid: int) -> pd.DataFrame:
    """Most significant distinct (rsID, trait) pairs, at most `per_rsid` per rsID."""
    ranked = report.dropna(subset=["neg_log10_p"])
    for flag in ("multi_snp_haplotype", "snp_interaction"):
        if flag in ranked:
            ranked = ranked[ranked[flag] != True]  # noqa: E712 - column may hold NaN
    ranked = (
        ranked.sort_values("neg_log10_p", ascending=False)
        .drop_duplicates(["rsid", "trait_label"])
        .groupby("rsid", sort=False)
        .head(per_rsid)
        .head(top)
    )
    return ranked.iloc[::-1]  # largest at the top of a horizontal bar chart


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_report(report: pd.DataFrame, output: Path, top: int = 20, per_rsid: int = 3) -> bool:
    """Save the two-panel plot. Returns False (and writes nothing) if there is no data."""
    missing = REQUIRED - set(report.columns)
    if missing:
        raise ValueError(
            f"GWAS report is missing columns: {', '.join(sorted(missing))}. "
            "Re-run traits/gwas_catalog.py to regenerate it."
        )
    report = report.dropna(subset=["rsid"]).copy()
    if report.empty:
        return False

    report["trait_label"] = trait_label(report)
    report["neg_log10_p"] = pd.to_numeric(report["neg_log10_p"], errors="coerce")
    copies = pd.to_numeric(report["effect_allele_copies"], errors="coerce")
    ambiguous = report["strand_ambiguous"].astype(str).str.lower() == "true"
    report["copies_shown"] = copies.where(~ambiguous)
    report["effect_allele"] = report["effect_allele"].fillna("").astype(str)

    trait_counts = report.groupby("rsid")["trait_label"].nunique().sort_values()
    genotype_by_rsid = report.groupby("rsid")["genotype"].first()
    ranked = top_associations(report, top, per_rsid)

    height = max(6, 2.7 + 0.34 * max(len(trait_counts), len(ranked), 1))
    fig, (count_ax, pvalue_ax) = plt.subplots(
        1, 2, figsize=(15, height), gridspec_kw={"width_ratios": [1, 2.2]},
        facecolor=SURFACE,
    )
    style_axes(count_ax)
    style_axes(pvalue_ax)

    count_labels = [f"{rsid} ({genotype_by_rsid[rsid]})" for rsid in trait_counts.index]
    count_ax.barh(count_labels, trait_counts.values, color=NEUTRAL_BAR, height=0.7)
    count_ax.set_title("Distinct traits per rsID", loc="left", color=INK)
    count_ax.set_xlabel("Distinct EFO traits (your genotype in brackets)", color=MUTED)

    if len(ranked):
        labels = [
            textwrap.shorten(f"{row.rsid}-{row.effect_allele or '?'}: {row.trait_label}",
                             width=72, placeholder="...")
            for row in ranked.itertuples()
        ]
        colors = [COPY_COLORS.get(c, NEUTRAL_BAR) if pd.notna(c) else NEUTRAL_BAR
                  for c in ranked["copies_shown"]]
        positions = np.arange(len(ranked))
        pvalue_ax.barh(positions, ranked["neg_log10_p"], color=colors, height=0.7)
        pvalue_ax.set_yticks(positions, labels)
        pvalue_ax.set_xscale("log")
        pvalue_ax.set_xlim(left=1)
        pvalue_ax.axvline(GENOME_WIDE, color=THRESHOLD, linestyle="--", linewidth=1)
        handles = [Patch(color=COPY_COLORS[k], label=COPY_LABELS[k]) for k in COPY_COLORS]
        handles += [
            Patch(color=NEUTRAL_BAR, label=UNKNOWN_LABEL),
            Line2D([], [], color=THRESHOLD, linestyle="--", linewidth=1,
                   label="Genome-wide threshold (5e-8)"),
        ]
        pvalue_ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK,
                         title="Your copies of the effect allele", title_fontsize=8,
                         loc="lower right")
    else:
        pvalue_ax.text(
            0.5, 0.5, "No parseable p-values", ha="center", va="center",
            transform=pvalue_ax.transAxes, color=MUTED,
        )
    pvalue_ax.set_title(
        f"Top distinct trait associations (max {per_rsid} per rsID)", loc="left", color=INK,
    )
    pvalue_ax.set_xlabel("-log10(p-value), log scale", color=MUTED)

    fig.suptitle(
        "GWAS Catalog associations: statistical significance, not personal risk",
        x=0.06, ha="left", color=INK,
    )
    fig.text(
        0.06, 0.005,
        "Grey also covers A/T and C/G SNPs, where the strand of the catalog allele is ambiguous.",
        color=MUTED, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return True


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("--input", type=Path, help="GWAS CSV (default: configured outputs folder)")
    parser.add_argument("--output", type=Path, help="PNG path (default: configured outputs folder)")
    parser.add_argument("--top", type=int, default=20, help="number of distinct associations to show")
    parser.add_argument("--per-rsid", type=int, default=3,
                        help="maximum associations shown per rsID")
    args = parser.parse_args()
    if args.top < 1 or args.per_rsid < 1:
        parser.error("--top and --per-rsid must be at least 1")

    cfg = load_config(args.config)
    input_path = args.input or cfg["out_dir"] / "gwas_catalog_associations.csv"
    output_path = args.output or cfg["out_dir"] / "gwas_catalog_associations.png"
    report = pd.read_csv(input_path)
    if plot_report(report, output_path, args.top, args.per_rsid):
        print(f"Plot saved to {output_path}")
    else:
        print(f"No associations in {input_path}; nothing to plot")


if __name__ == "__main__":
    main()
