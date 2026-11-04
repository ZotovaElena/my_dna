"""Basic QC: call rate, autosomal heterozygosity, sex inference from X."""

import json

import pandas as pd

from common import AUTOSOMES, config_arg, is_het, load_config, load_myheritage, read_header

# GRCh37 pseudo-autosomal regions; excluded so males show ~0 X heterozygosity
PARS = [(60_001, 2_699_520), (154_931_044, 155_260_560)]


def infer_sex(het_x: float, het_auto: float) -> str:
    ratio = het_x / het_auto
    if ratio < 0.2:
        return "male"
    return "female" if ratio > 0.4 else "uncertain"


def main() -> None:
    args = config_arg(__doc__).parse_args()
    cfg = load_config(args.config)

    df = load_myheritage(cfg["raw_file"], drop_nocalls=False)
    called = df["genotype"].str.fullmatch(r"[ACGT]{1,2}", na=False)
    calls = df[called]
    auto = calls[calls["chrom"].isin(AUTOSOMES)]
    x = calls[calls["chrom"] == "X"]
    x = x[~pd.concat([x["pos"].between(a, b) for a, b in PARS], axis=1).any(axis=1)]

    het_auto = is_het(auto["genotype"]).mean()
    het_x = is_het(x["genotype"]).mean() if len(x) else float("nan")
    per_chrom = (
        calls.assign(het=is_het(calls["genotype"]))
        .groupby("chrom")["het"].agg(n="size", het_rate="mean")
        .reindex(AUTOSOMES + ["X", "Y", "MT"]).dropna()
        .astype({"n": int})
    )

    report = {
        "header": read_header(cfg["raw_file"]),
        "n_variants": len(df),
        "call_rate": round(called.mean(), 5),
        "het_autosomal": round(het_auto, 4),
        "het_x_nonpar": round(het_x, 4),
        "inferred_sex": infer_sex(het_x, het_auto) if len(x) else "no X calls",
        "has_y": bool((calls["chrom"] == "Y").any()),
        "has_mt": bool((calls["chrom"] == "MT").any()),
        "per_chrom": per_chrom.round(4).reset_index().to_dict(orient="records"),
    }
    with open(cfg["out_dir"] / "qc.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "per_chrom"}, indent=2))
    print(per_chrom.round(4).to_string())


if __name__ == "__main__":
    main()
