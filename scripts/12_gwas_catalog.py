"""Query GWAS Catalog associations for selected rsIDs in a MyHeritage DNA file.

Only rsIDs are sent to the public API; genotype calls remain local.
GWAS associations are population-level findings, not personal medical predictions.

Raw API responses are cached per rsID under <out_dir>/gwas_cache/; pass --refresh
to download them again.
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common import config_arg, load_config, load_myheritage

API_URL = "https://www.ebi.ac.uk/gwas/rest/api/v2/associations"
PAGE_SIZE = 200
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
PALINDROMIC = ({"A", "T"}, {"C", "G"})

COLUMNS = [
    "rsid", "genotype", "effect_allele", "effect_allele_copies", "strand_ambiguous",
    "efo_trait", "efo_id", "reported_trait", "neg_log10_p", "pvalue",
    "pvalue_mantissa", "pvalue_exponent", "pvalue_description",
    "or_value", "beta", "ci_lower", "ci_upper", "range", "risk_frequency",
    "mapped_genes", "accession_id", "pubmed_id", "first_author",
    "multi_snp_haplotype", "snp_interaction",
]


def make_session() -> requests.Session:
    """HTTP session that retries transient failures with exponential backoff."""
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["Accept"] = "application/json"
    return session


def fetch_associations(session: requests.Session, rsid: str) -> list[dict]:
    """Fetch every association page for one rsID."""
    associations: list[dict] = []
    page_number = 0
    while True:
        response = session.get(
            API_URL,
            params={"rs_id": rsid, "size": PAGE_SIZE, "page": page_number},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        page = data.get("page") if isinstance(data, dict) else None
        if not isinstance(page, dict) or "totalPages" not in page:
            raise ValueError(f"Unexpected GWAS Catalog response for {rsid}: no page info")
        if page["totalElements"] == 0:
            return []
        batch = data.get("_embedded", {}).get("associations")
        if not isinstance(batch, list):
            raise ValueError(f"Unexpected GWAS Catalog response for {rsid}: no associations")
        for item in batch:
            item.pop("_links", None)
        associations.extend(batch)
        page_number += 1
        if page_number >= page["totalPages"]:
            break
    if len(associations) != page["totalElements"]:
        raise ValueError(
            f"{rsid}: expected {page['totalElements']} associations, got {len(associations)}"
        )
    return associations


def cached_associations(
    session: requests.Session, rsid: str, cache_dir: Path, refresh: bool
) -> list[dict]:
    """Return associations for an rsID, reading or writing the local JSON cache."""
    cache_file = cache_dir / f"{rsid}.json"
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())
    associations = fetch_associations(session, rsid)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(associations))
    return associations


def neg_log10_p(association: dict) -> float | None:
    """Exact -log10(p) from mantissa/exponent, so tiny p-values don't underflow to 0."""
    mantissa = association.get("pvalue_mantissa")
    exponent = association.get("pvalue_exponent")
    if isinstance(mantissa, (int, float)) and isinstance(exponent, (int, float)) and mantissa > 0:
        return -(math.log10(mantissa) + exponent)
    p_value = association.get("p_value")
    if isinstance(p_value, (int, float)) and p_value > 0:
        return -math.log10(p_value)
    return None


def effect_allele(association: dict, rsid: str) -> str:
    """Effect (risk) allele the catalog reports for this rsID; '' if unknown."""
    alleles = [
        entry.get("effect_allele", "")
        for entry in association.get("snp_allele") or []
        if entry.get("rs_id") == rsid
    ]
    alleles = sorted({a.strip().upper() for a in alleles if a})
    return "/".join(alleles)


def allele_copies(genotype: str | None, allele: str) -> int | None:
    """Copies of a single-base effect allele in the genotype; None if undeterminable."""
    if not genotype or allele not in COMPLEMENT:
        return None
    return genotype.count(allele)


def is_strand_ambiguous(genotype: str | None, allele: str) -> bool:
    """True when the alleles involved are A/T or C/G, so strand can't be inferred."""
    if not genotype or allele not in COMPLEMENT:
        return False
    return set(genotype) | {allele} in PALINDROMIC


def join(values) -> str:
    return "; ".join(str(v) for v in values or [] if v not in (None, ""))


def association_row(association: dict, rsid: str, genotype: str | None) -> dict:
    allele = effect_allele(association, rsid)
    efo = association.get("efo_traits") or []
    return {
        "rsid": rsid,
        "genotype": genotype or "not genotyped",
        "effect_allele": allele,
        "effect_allele_copies": allele_copies(genotype, allele),
        "strand_ambiguous": is_strand_ambiguous(genotype, allele),
        "efo_trait": join(t.get("efo_trait") for t in efo),
        "efo_id": join(t.get("efo_id") for t in efo),
        "reported_trait": join(association.get("reported_trait")),
        "neg_log10_p": neg_log10_p(association),
        "pvalue": association.get("p_value"),
        "pvalue_mantissa": association.get("pvalue_mantissa"),
        "pvalue_exponent": association.get("pvalue_exponent"),
        "pvalue_description": association.get("pvalue_description") or "",
        "or_value": association.get("or_value", association.get("or_per_copy_num")),
        "beta": association.get("beta") or "",
        "ci_lower": association.get("ci_lower"),
        "ci_upper": association.get("ci_upper"),
        "range": association.get("range") or "",
        "risk_frequency": association.get("risk_frequency"),
        "mapped_genes": join(association.get("mapped_genes")),
        "accession_id": association.get("accession_id"),
        "pubmed_id": association.get("pubmed_id"),
        "first_author": association.get("first_author"),
        "multi_snp_haplotype": association.get("multi_snp_haplotype"),
        "snp_interaction": association.get("snp_interaction"),
    }


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument(
        "--rsid",
        nargs="+",
        action="append",
        help="rsID(s) to query; repeat the option or pass several IDs. "
             "Defaults to rsIDs in the configured traits list.",
    )
    parser.add_argument("--refresh", action="store_true", help="ignore the local API cache")
    args = parser.parse_args()
    cfg = load_config(args.config)

    genotypes = load_myheritage(cfg["raw_file"]).set_index("rsid")["genotype"].to_dict()
    requested = [rsid for group in args.rsid for rsid in group] if args.rsid else list(cfg["traits"])
    rsids = list(dict.fromkeys(requested))
    cache_dir = cfg["out_dir"] / "gwas_cache"

    session = make_session()
    rows, failed = [], []
    for rsid in rsids:
        try:
            associations = cached_associations(session, rsid, cache_dir, args.refresh)
        except (requests.RequestException, ValueError) as exc:
            print(f"WARNING: {rsid} failed: {exc}", file=sys.stderr)
            failed.append(rsid)
            continue
        print(f"{rsid}: {len(associations)} associations")
        rows.extend(association_row(a, rsid, genotypes.get(rsid)) for a in associations)

    table = pd.DataFrame(rows, columns=COLUMNS)
    table["effect_allele_copies"] = table["effect_allele_copies"].astype("Int64")
    output = cfg["out_dir"] / "gwas_catalog_associations.csv"
    table.to_csv(output, index=False)
    print(f"Associations: {len(table)} across {len(rsids) - len(failed)} rsIDs")
    print(f"Saved to {output}")
    if failed:
        print(f"Failed rsIDs (rerun to retry): {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
