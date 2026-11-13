"""Map GWAS Catalog associations to trait categories and plot what relates to your alleles.

Reads the CSV from 12_gwas_catalog.py and the GWAS Catalog EFO trait-to-category
mapping (downloaded once into <ref_dir>/gwas_catalog/; no personal data is sent).

Outputs (in out_dir):
  gwas_trait_map.csv        every association with its category and effect direction
  gwas_carried_traits.csv   genome-wide associations whose effect allele you carry
  gwas_trait_overview.png   SNP x trait-category grid
  gwas_carried_traits.png   top carried-allele traits per category

These are population-level associations, not personal risk estimates.
"""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import requests

from common import config_arg, load_config

MAPPING_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/gwas-efo-trait-mappings.tsv"
)
GENOME_WIDE = -np.log10(5e-8)
UNCATEGORISED = "Uncategorised"

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#3987e5", "#0d366b"])
COPY_COLORS = {1: "#3987e5", 2: "#184f95"}  # ordinal blue steps 400 / 600
DIRECTION_MARKERS = {"higher": "^", "lower": "v", "unknown": "o"}


# ---------------------------------------------------------------- annotation


def load_mapping(ref_dir: Path, refresh: bool) -> pd.DataFrame:
    """Download (once) and read the GWAS Catalog EFO trait mapping table."""
    path = ref_dir / "gwas_catalog" / "gwas-efo-trait-mappings.tsv"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading trait mapping to {path}")
        response = requests.get(MAPPING_URL, timeout=300)
        response.raise_for_status()
        path.write_bytes(response.content)
    mapping = pd.read_csv(path, sep="\t", dtype=str)
    expected = {"Disease trait", "EFO term", "EFO URI", "Parent term"}
    if missing := expected - set(mapping.columns):
        raise ValueError(f"Trait mapping file is missing columns: {sorted(missing)}")
    return mapping


def category_lookups(mapping: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Category lookups by EFO id, EFO term name and reported trait name."""
    mapping = mapping.dropna(subset=["Parent term"])
    mapping = mapping[mapping["Parent term"] != "NR"]
    efo_id = mapping["EFO URI"].str.rsplit("/", n=1).str[-1]

    def lookup(keys: pd.Series) -> dict:
        return dict(zip(keys, mapping["Parent term"]))

    return (
        lookup(efo_id),
        lookup(mapping["EFO term"].str.lower()),
        lookup(mapping["Disease trait"].str.lower()),
    )


def split(value) -> list[str]:
    return [part.strip() for part in str(value).split(";") if part.strip()] if pd.notna(value) else []


def assign_category(row, by_id: dict, by_term: dict, by_reported: dict) -> str:
    for efo_id in split(row.efo_id):
        if efo_id in by_id:
            return by_id[efo_id]
    for name in split(row.efo_trait):
        if name.lower() in by_term:
            return by_term[name.lower()]
    for name in split(row.reported_trait):
        if name.lower() in by_reported:
            return by_reported[name.lower()]
    return UNCATEGORISED


def effect_direction(row) -> str:
    """Direction of the effect allele: from the OR if present, else the beta text."""
    odds_ratio = pd.to_numeric(row.or_value, errors="coerce")
    if pd.notna(odds_ratio) and odds_ratio != 1:
        return "higher" if odds_ratio > 1 else "lower"
    beta = str(row.beta).lower() if pd.notna(row.beta) else ""
    if "increase" in beta:
        return "higher"
    if "decrease" in beta:
        return "lower"
    return "unknown"


SD_HINTS = ("inv-norm", "inverse norm", "standardi", "z-score", "z score", " sd", "(sd", "rank-based")


def effect_size(row) -> tuple[str, str]:
    """(tier, text) for the per-allele effect: tier is very small/small/moderate/large/unknown."""
    odds_ratio = pd.to_numeric(row.or_value, errors="coerce")
    if pd.notna(odds_ratio) and odds_ratio > 0:
        fold = max(odds_ratio, 1 / odds_ratio)
        tier = ("very small" if fold < 1.1 else "small" if fold < 1.5
                else "moderate" if fold < 3 else "large")
        return tier, f"odds ratio {odds_ratio:.2f} per copy"
    beta_text = str(row.beta).strip() if pd.notna(row.beta) else ""
    beta = pd.to_numeric(beta_text.split(" ")[0], errors="coerce") if beta_text else np.nan
    if pd.isna(beta):
        return "unknown", ""
    context = f" {beta_text} {row.reported_trait}".lower()
    if any(hint in context for hint in SD_HINTS):
        size = abs(beta)
        tier = ("very small" if size < 0.05 else "small" if size < 0.2
                else "moderate" if size < 0.5 else "large")
        return tier, f"{abs(beta):.3g} standard deviations per copy"
    units = beta_text.split(" ", 1)[1] if " " in beta_text else ""
    units = units.replace("increase", "").replace("decrease", "").strip() or "units"
    return "unknown", f"{abs(beta):.3g} {units} per copy (study-specific units)"


def is_disease(category: str) -> bool:
    return any(word in category.lower() for word in ("disease", "disorder", "cancer"))


def trait_phrase(trait: str, category: str, direction: str) -> str:
    trait = trait.removesuffix(" measurement")
    if is_disease(category):
        return f"{direction} risk of {trait}"
    return f"{direction} {trait} values"


def interpret(row) -> tuple[str, str]:
    """('toward higher' | 'toward lower' | 'unclear', plain-language sentence)."""
    allele = row.effect_allele if isinstance(row.effect_allele, str) and row.effect_allele else ""
    copies = row.effect_allele_copies
    genotype = row.genotype
    tier, size_text = row.effect_size_tier, row.effect_size_text
    linked = trait_phrase(row.trait_label, row.category, row.direction)
    size_note = (f" The effect is {tier} ({size_text})." if tier != "unknown" and size_text
                 else f" Effect: {size_text}." if size_text else "")

    if genotype == "not genotyped":
        return "unclear", "This SNP is not in your DNA file, so this result can't be applied to you."
    if str(row.strand_ambiguous).lower() == "true":
        return "unclear", (f"This is an A/T or C/G SNP, so it's unclear whether the catalog's allele "
                           f"{allele} matches the strand of your genotype ({genotype}).")
    if not allele or pd.isna(copies):
        return "unclear", ("The study did not report which allele carries the effect, "
                           "so this can't be matched to your genotype.")
    if row.direction == "unknown":
        return "unclear", (f"You carry {int(copies)} cop{'y' if copies == 1 else 'ies'} of {allele}, "
                           f"but the study did not report the direction of its effect.")

    opposite = "lower" if row.direction == "higher" else "higher"
    if copies >= 1:
        verdict = f"toward {row.direction}"
        text = (f"You carry {int(copies)} cop{'y' if copies == 1 else 'ies'} of {allele} "
                f"(genotype {genotype}). In this study, {allele} was associated with "
                f"{linked}.")
    else:
        verdict = f"toward {opposite}"
        text = (f"You don't carry {allele} (genotype {genotype}). In this study {allele} was "
                f"associated with {linked}, so compared with carriers your genotype points "
                f"toward {trait_phrase(row.trait_label, row.category, opposite)}.")
    return verdict, text + size_note


def annotate(report: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    by_id, by_term, by_reported = category_lookups(mapping)
    report = report.copy()
    report["trait_label"] = report["efo_trait"].where(
        report["efo_trait"].fillna("").str.strip() != "", report["reported_trait"]
    ).fillna("Unlabeled trait")
    report["category"] = [assign_category(r, by_id, by_term, by_reported)
                          for r in report.itertuples()]
    report["direction"] = [effect_direction(r) for r in report.itertuples()]
    report["neg_log10_p"] = pd.to_numeric(report["neg_log10_p"], errors="coerce")
    report["effect_allele_copies"] = pd.to_numeric(report["effect_allele_copies"],
                                                   errors="coerce")
    sizes = [effect_size(r) for r in report.itertuples()]
    report["effect_size_tier"] = [tier for tier, _ in sizes]
    report["effect_size_text"] = [text for _, text in sizes]
    verdicts = [interpret(r) for r in report.itertuples()]
    report["for_you"] = [verdict for verdict, _ in verdicts]
    report["explanation"] = [text for _, text in verdicts]
    return report


def carried_traits(report: pd.DataFrame) -> pd.DataFrame:
    """Best genome-wide association per (rsID, trait) whose effect allele you carry."""
    carried = report[
        (report["effect_allele_copies"] >= 1)
        & (report["neg_log10_p"] >= GENOME_WIDE)
        & (report["strand_ambiguous"].astype(str).str.lower() != "true")
        & (report["multi_snp_haplotype"].astype(str).str.lower() != "true")
        & (report["snp_interaction"].astype(str).str.lower() != "true")
    ]
    return (
        carried.sort_values("neg_log10_p", ascending=False)
        .drop_duplicates(["rsid", "trait_label"])
        .reset_index(drop=True)
    )


def main_gene(genes: pd.Series) -> str:
    values = genes.dropna().astype(str)
    values = values[values != ""]
    return values.mode().iat[0] if len(values) else ""


# ---------------------------------------------------------------- plotting


def style_axes(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_overview(report: pd.DataFrame, output: Path) -> None:
    """Grid: dot size = distinct traits, dot shade = strongest -log10(p)."""
    cells = (
        report.groupby(["category", "rsid"])
        .agg(traits=("trait_label", "nunique"), best=("neg_log10_p", "max"))
        .reset_index()
    )
    category_order = (report.groupby("category")["trait_label"].nunique()
                      .sort_values().index.tolist())
    rsid_order = (report.groupby("rsid")["trait_label"].nunique()
                  .sort_values(ascending=False).index.tolist())
    snp_info = report.groupby("rsid").agg(genotype=("genotype", "first"),
                                          gene=("mapped_genes", main_gene))
    x = cells["rsid"].map({r: i for i, r in enumerate(rsid_order)})
    y = cells["category"].map({c: i for i, c in enumerate(category_order)})
    size_scale = 900 / max(cells["traits"].max(), 1)

    fig, ax = plt.subplots(
        figsize=(max(8, 1.0 * len(rsid_order) + 4), 0.5 * len(category_order) + 3),
        facecolor=SURFACE,
    )
    style_axes(ax, grid_axis="both")
    best = cells["best"].clip(lower=1).fillna(1)
    norm = LogNorm(vmin=max(best.min(), 1), vmax=max(best.max(), 10))
    points = ax.scatter(x, y, s=cells["traits"] * size_scale + 15, c=best, cmap=SEQUENTIAL,
                        norm=norm, edgecolors=SURFACE, linewidths=2, zorder=3)

    ax.set_xticks(range(len(rsid_order)), [
        f"{r}\n{snp_info.at[r, 'gene']}\n{snp_info.at[r, 'genotype']}" for r in rsid_order
    ])
    ax.set_yticks(range(len(category_order)), category_order)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(-0.6, len(rsid_order) - 0.4)
    ax.set_ylim(-0.6, len(category_order) - 0.4)

    colorbar = fig.colorbar(points, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Strongest -log10(p) in cell", color=MUTED, fontsize=8)
    colorbar.ax.tick_params(colors=MUTED, labelsize=7)
    colorbar.outline.set_visible(False)
    reference = sorted({1, max(1, round(cells["traits"].max() / 4)), int(cells["traits"].max())})
    handles = [
        plt.scatter([], [], s=n * size_scale + 15, color="#9ec5f4", edgecolors=SURFACE,
                    label=f"{n} trait{'s' if n > 1 else ''}")
        for n in reference
    ]
    ax.legend(handles=handles, title="Distinct traits", title_fontsize=8, fontsize=8,
              frameon=False, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.12, 1),
              labelspacing=2.4, handletextpad=2.0, borderpad=1.2)

    ax.set_title("Which kinds of traits each SNP is associated with",
                 loc="left", color=INK, pad=12)
    fig.text(0.01, 0.005, "Column labels: rsID, main mapped gene, your genotype. "
             "GWAS Catalog associations, not personal risk.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_carried(carried: pd.DataFrame, output: Path, per_category: int) -> None:
    """Top traits per category for effect alleles you carry, with effect direction."""
    if carried.empty:
        return
    category_order = (carried.groupby("category")["neg_log10_p"].max()
                      .sort_values(ascending=False).index.tolist())

    rows = []  # (kind, payload) from top to bottom
    for category in category_order:
        group = carried[carried["category"] == category]
        rows.append(("header", f"{category} ({len(group)} trait{'s' if len(group) > 1 else ''})"))
        rows.extend(("trait", row) for row in group.head(per_category).itertuples())

    fig, ax = plt.subplots(figsize=(12, 0.27 * len(rows) + 2.2), facecolor=SURFACE)
    style_axes(ax)
    ax.set_xscale("log")
    ax.set_xlim(GENOME_WIDE * 0.8, carried["neg_log10_p"].max() * 1.6)
    ax.set_ylim(len(rows) - 0.5, -0.8)
    ax.set_yticks([])

    for i, (kind, item) in enumerate(rows):
        if kind == "header":
            ax.text(-0.01, i, item, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", color=INK, fontsize=9, fontweight="bold")
            continue
        genes = f" ({item.mapped_genes})" if pd.notna(item.mapped_genes) and item.mapped_genes else ""
        label = textwrap.shorten(f"{item.trait_label} | {item.rsid}-{item.effect_allele}{genes}",
                                 width=80, placeholder="...")
        ax.text(-0.01, i, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", color=MUTED, fontsize=8)
        ax.hlines(i, GENOME_WIDE * 0.8, item.neg_log10_p, color=GRID, linewidth=1, zorder=1)
        ax.scatter(item.neg_log10_p, i, s=70, marker=DIRECTION_MARKERS[item.direction],
                   color=COPY_COLORS.get(int(item.effect_allele_copies), "#3987e5"),
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)

    handles = [
        Line2D([], [], marker="s", linestyle="", markersize=8, color=COPY_COLORS[1],
               label="You carry 1 copy"),
        Line2D([], [], marker="s", linestyle="", markersize=8, color=COPY_COLORS[2],
               label="You carry 2 copies"),
        Line2D([], [], marker="^", linestyle="", markersize=8, color=MUTED,
               label="Allele linked to higher value / risk"),
        Line2D([], [], marker="v", linestyle="", markersize=8, color=MUTED,
               label="Allele linked to lower value / risk"),
        Line2D([], [], marker="o", linestyle="", markersize=8, color=MUTED,
               label="Direction not reported"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK,
              loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3)
    ax.set_xlabel("-log10(p-value), log scale (all genome-wide significant)", color=MUTED)
    fig.suptitle("Traits associated with alleles you carry "
                 f"(top {per_category} per category)", x=0.02, ha="left", color=INK)
    fig.text(0.02, 0.005, "Population-level GWAS associations; effect sizes per allele are "
             "usually small. Full list: gwas_carried_traits.csv.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("--input", type=Path, help="GWAS CSV (default: configured outputs folder)")
    parser.add_argument("--per-category", type=int, default=5,
                        help="traits shown per category in the carried-traits plot")
    parser.add_argument("--refresh-mapping", action="store_true",
                        help="download the trait mapping file again")
    args = parser.parse_args()
    if args.per_category < 1:
        parser.error("--per-category must be at least 1")

    cfg = load_config(args.config)
    out_dir = cfg["out_dir"]
    input_path = args.input or out_dir / "gwas_catalog_associations.csv"
    report = pd.read_csv(input_path)
    if report.empty:
        print(f"No associations in {input_path}; nothing to do")
        return
    if "efo_id" not in report.columns:
        raise ValueError("Old-format GWAS CSV; re-run 12_gwas_catalog.py first.")

    report = annotate(report, load_mapping(cfg["ref_dir"], args.refresh_mapping))
    carried = carried_traits(report)
    report.to_csv(out_dir / "gwas_trait_map.csv", index=False)
    carried.to_csv(out_dir / "gwas_carried_traits.csv", index=False)

    plot_overview(report, out_dir / "gwas_trait_overview.png")
    plot_carried(carried, out_dir / "gwas_carried_traits.png", args.per_category)

    uncategorised = (report["category"] == UNCATEGORISED).sum()
    print(f"Associations: {len(report)} ({uncategorised} uncategorised), "
          f"categories: {report['category'].nunique()}")
    print(f"Carried genome-wide traits: {len(carried)} across {carried['rsid'].nunique()} rsIDs")
    print(f"Saved CSVs and plots to {out_dir}")


if __name__ == "__main__":
    main()
