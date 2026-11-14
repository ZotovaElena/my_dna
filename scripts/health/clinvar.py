"""Annotate genotypes with ClinVar (GRCh37) by position + allele.

Low-pass/imputed and array calls of rare variants are often false positives:
treat every pathogenic hit as a hypothesis to confirm with a clinical test.
"""

import csv
import gzip
import itertools

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import alt_copies, chrom_key, config_arg, load_config, load_myheritage

INFO_KEYS = ("CLNSIG", "CLNDN", "CLNREVSTAT", "GENEINFO")
STARS = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,
}
PATHOGENIC = r"^(Pathogenic|Likely_pathogenic|Pathogenic/Likely_pathogenic)"


def parse_info(info: str) -> dict[str, str]:
    fields = dict(kv.split("=", 1) for kv in info.split(";") if "=" in kv)
    return {k: fields.get(k, "") for k in INFO_KEYS}


def count_header_lines(path) -> int:
    with gzip.open(path, "rt") as f:
        return sum(1 for _ in itertools.takewhile(lambda line: line.startswith("#"), f))


def load_clinvar_at(path, sites: pd.DataFrame, chunksize: int = 500_000) -> pd.DataFrame:
    """Stream ClinVar VCF and keep SNVs at the sample's positions."""
    cols = ["chrom", "pos", "clinvar_id", "ref", "alt", "info"]
    keep = []
    # Skip the header by line count, not comment="#": INFO values contain '#'
    # (e.g. UniProtKB:Q9Y6L6#VAR_015076), and comment= would truncate those records.
    for chunk in pd.read_csv(path, sep="\t", skiprows=count_header_lines(path), header=None,
                             usecols=[0, 1, 2, 3, 4, 7], names=cols, quoting=csv.QUOTE_NONE,
                             dtype={"chrom": str, "pos": int}, chunksize=chunksize):
        chunk = chunk[(chunk["ref"].str.len() == 1) & (chunk["alt"].str.len() == 1)]
        keep.append(chunk.merge(sites, on=["chrom", "pos"]))
    return pd.concat(keep, ignore_index=True)


def main() -> None:
    args = config_arg(__doc__).parse_args()
    cfg = load_config(args.config)

    sample = load_myheritage(cfg["raw_file"])
    hits = load_clinvar_at(cfg["ref_dir"] / "clinvar.vcf.gz", sample[["chrom", "pos", "rsid", "genotype"]])
    # None (→ NaN) where the genotype isn't made of ClinVar's REF/ALT alleles
    hits["alt_copies"] = [alt_copies(g, r, a) for g, r, a in zip(hits["genotype"], hits["ref"], hits["alt"])]
    hits["allele_mismatch"] = hits["alt_copies"].isna()
    carriers = hits[hits["alt_copies"] > 0].astype({"alt_copies": int})

    info = pd.DataFrame([parse_info(i) for i in carriers["info"]], index=carriers.index)
    carriers = carriers.drop(columns="info").join(info)
    carriers["stars"] = carriers["CLNREVSTAT"].map(STARS).fillna(0).astype(int)
    carriers["gene"] = carriers["GENEINFO"].str.split(":").str[0]
    carriers = carriers.sort_values(["stars", "chrom", "pos"], ascending=[False, True, True], key=chrom_key)
    carriers.to_csv(cfg["out_dir"] / "clinvar_carriers.csv", index=False)

    # Records without CLNSIG have no germline classification (e.g. somatic/oncogenicity-only submissions).
    carriers["CLNSIG"] = carriers["CLNSIG"].replace("", "(no germline classification)")
    carriers["conditions"] = carriers["CLNDN"].map(clean_conditions)

    print(f"ClinVar SNVs at sample sites: {len(hits)} | carrying ALT: {len(carriers)} "
          f"| allele mismatches: {hits['allele_mismatch'].sum()}")
    print(carriers["CLNSIG"].value_counts().head(12).to_string())

    masks = {
        "Pathogenic / likely pathogenic": carriers["CLNSIG"].str.match(PATHOGENIC),
        "Drug response": carriers["CLNSIG"].str.contains("drug_response"),
        "Risk factor": carriers["CLNSIG"].str.contains("risk_factor"),
    }
    sections = [(label, carriers[m].sort_values(["stars", "gene"], ascending=[False, True]))
                for label, m in masks.items()]
    for label, sub in sections:
        print(f"\n== {label}: {len(sub)} ==")
        if len(sub):
            table = sub[TERM_COLS].assign(stars=sub["stars"].map(star_str))
            print(table.head(40).to_string(index=False, max_colwidth=60))

    report = cfg["out_dir"] / "clinvar_report.html"
    report.write_text(html_report(carriers, sections, len(hits)), encoding="utf-8")
    print(f"\nFull, filterable report: {report}")


TERM_COLS = ["rsid", "gene", "genotype", "alt_copies", "stars", "CLNSIG", "conditions"]
NOISE = {"not_provided", "not_specified", ".", ""}


def clean_conditions(clndn: str) -> str:
    """'not_provided|Warfarin_response_-_Dosage' -> 'Warfarin response - Dosage'."""
    names = [n.replace("_", " ") for n in str(clndn).split("|") if n not in NOISE]
    return "; ".join(dict.fromkeys(names)) or "-"


def star_str(n: int) -> str:
    return "★" * n + "☆" * (4 - n)


def html_report(carriers: pd.DataFrame, sections: list[tuple[str, pd.DataFrame]], n_sites: int) -> str:
    """Self-contained HTML (no external resources): one sortable, filterable table per section.

    `carriers` and each section's rows arrive already sorted.
    """
    from html import escape

    cols = ["rsid", "gene", "chrom", "pos", "genotype", "ref", "alt", "alt_copies", "stars",
            "CLNSIG", "conditions", "clinvar_id"]

    def table(df: pd.DataFrame) -> str:
        head = "".join(f"<th>{c}</th>" for c in cols)
        rows = []
        for r in df[cols].itertuples(index=False):
            d = r._asdict()
            cells = {c: escape(str(d[c])) for c in cols}
            cells["rsid"] = f'<a href="https://www.ncbi.nlm.nih.gov/snp/{cells["rsid"]}" target="_blank">{cells["rsid"]}</a>'
            cells["clinvar_id"] = (f'<a href="https://www.ncbi.nlm.nih.gov/clinvar/variation/{cells["clinvar_id"]}/" '
                                   f'target="_blank">{cells["clinvar_id"]}</a>')
            cells["stars"] = f'<span data-sort="{d["stars"]}">{star_str(d["stars"])}</span>'
            rows.append("<tr>" + "".join(f"<td>{cells[c]}</td>" for c in cols) + "</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    parts = [f"<h2>{escape(label)} <small>({len(sub)})</small></h2>"
             + (table(sub) if len(sub) else "<p>None.</p>")
             for label, sub in sections]
    counts = carriers["CLNSIG"].value_counts()
    summary = "".join(f"<tr><td>{escape(k)}</td><td>{v}</td></tr>" for k, v in counts.items())

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>ClinVar carriers</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;color:#222;background:#fff}}
h1{{margin:0 0 4px}} h2{{margin-top:32px}} small{{color:#777;font-weight:normal}}
.note{{background:#fff7e0;border-left:4px solid #e0a800;padding:8px 12px;max-width:900px}}
input{{font-size:14px;padding:6px 10px;width:360px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;margin-top:6px}}
th,td{{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f3f3f3;cursor:pointer;position:sticky;top:0;user-select:none}}
td:nth-child(11){{max-width:520px}} tr:hover td{{background:#f7fbff}}
.summary{{width:auto}} details{{margin-top:32px}}
</style></head><body>
<h1>ClinVar carriers</h1>
<p>{n_sites} ClinVar SNVs at your genotyped sites; you carry the ALT allele at {len(carriers)}.
Stars = ClinVar review status (★★★★ practice guideline … ☆☆☆☆ no assertion criteria).</p>
<p class="note">Exploratory only. Low-pass/imputed calls of rare variants are often false positives, and ClinVar
"risk factor"/"drug response" entries are mostly weak, population-level associations.
Confirm anything actionable with a clinical-grade test and a clinician or genetic counsellor.</p>
<input id="q" placeholder="Filter all tables (gene, rsid, condition, drug…)">
<table class="summary"><thead><tr><th>Clinical significance</th><th>Count</th></tr></thead><tbody>{summary}</tbody></table>
{''.join(parts)}
<details><summary><b>All ALT-carrying ClinVar sites ({len(carriers)})</b></summary>{table(carriers)}</details>
<script>
const q=document.getElementById('q');
q.addEventListener('input',()=>{{const t=q.value.toLowerCase();
  document.querySelectorAll('tbody tr').forEach(r=>{{r.style.display=r.textContent.toLowerCase().includes(t)?'':'none'}})}});
document.querySelectorAll('th').forEach(th=>th.addEventListener('click',()=>{{
  const tb=th.closest('table').tBodies[0],i=[...th.parentNode.children].indexOf(th),asc=th.dataset.asc!=='1';
  const key=td=>{{const s=td.querySelector('[data-sort]');const v=s?s.dataset.sort:td.textContent;return isNaN(v)||v===''?v.toLowerCase():+v}};
  [...tb.rows].sort((a,b)=>{{const x=key(a.cells[i]),y=key(b.cells[i]);return (x>y?1:x<y?-1:0)*(asc?1:-1)}}).forEach(r=>tb.appendChild(r));
  th.parentNode.querySelectorAll('th').forEach(h=>delete h.dataset.asc);th.dataset.asc=asc?'1':'0'}}));
</script></body></html>"""


if __name__ == "__main__":
    main()
