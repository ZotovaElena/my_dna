"""Single-SNP trait lookup (rsIDs from config) and optional APOE genotype."""

import pandas as pd

from common import config_arg, load_config, load_myheritage


def apoe(genotypes: dict[str, str]) -> str:
    """APOE from rs429358 (C = ε4) and rs7412 (T = ε2); ε2/ε4 is indistinguishable from rare ε1/ε3."""
    if not {"rs429358", "rs7412"} <= genotypes.keys():
        return "not genotyped"
    e4, e2 = genotypes["rs429358"].count("C"), genotypes["rs7412"].count("T")
    if e2 + e4 > 2:
        return "inconsistent calls"
    alleles = ["ε2"] * e2 + ["ε4"] * e4
    return "/".join(sorted(alleles + ["ε3"] * (2 - len(alleles))))


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("--apoe", action="store_true",
                        help="Also report APOE (Alzheimer's risk); off by default on purpose.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    genotypes = load_myheritage(cfg["raw_file"]).set_index("rsid")["genotype"].to_dict()
    rows = []
    for rsid, t in cfg["traits"].items():
        gt = genotypes.get(rsid)
        rows.append({
            "rsid": rsid,
            "trait": t["trait"],
            "genotype": gt or "not genotyped",
            "effect_allele": t["effect_allele"],
            "copies": gt.count(t["effect_allele"]) if gt else None,
            "effect": t["effect"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(cfg["out_dir"] / "traits.csv", index=False)
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(table.to_string(index=False))

    if args.apoe:
        print(f"\nAPOE: {apoe(genotypes)}")


if __name__ == "__main__":
    main()
