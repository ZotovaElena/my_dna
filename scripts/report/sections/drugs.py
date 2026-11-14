"""Drugs module: pharmacogenomics from PharmCAT."""

import html
import json
import re
from pathlib import Path

import pandas as pd

from report.core import card, details, esc, read_json, section, table, tile

PGX_SOURCES = ("CPIC", "DPWG", "FDA")


NOT_CALLED = {"", "n/a", "no result", "unknown", "unknown/unknown", "not called", "–"}


# A recommendation is actionable when it asks for a change, and is not a negation of one
PGX_ACTION = ("avoid", "alternative", "not recommended", "increas", "reduc", "decreas", "lower", "≤", "<=",
              "contraindicat", "do not use", "50%")


PGX_NEGATE = ("no need to avoid", "no reason to avoid", "genotype is available", "no recommendation",
              "no adjustments needed", "no action", "label recommended")


PGX_NO_CALL = {"no result", "uncertain susceptibility", "n/a", "indeterminate", ""}


def pgx_actionable(r: dict) -> bool:
    """Asks for a change, is not a negation, and rests on at least one real phenotype call."""
    text = r["rec"].lower()
    called = {p.strip().lower() for p in r["phenotype"].split(",")} - PGX_NO_CALL
    return bool(called) and any(k in text for k in PGX_ACTION) and not any(k in text for k in PGX_NEGATE)


STRENGTH = {"strong": 0, "moderate": 1, "optional": 2}


def _walk(o, path=(), up=()):
    """Every dict inside a JSON value, with its key path and its ancestor dicts."""
    if isinstance(o, dict):
        yield path, o, up
        for k, v in o.items():
            yield from _walk(v, path + (k,), up + (o,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, path + (i,), up)


def _source(path) -> str:
    return next((PGX_SOURCES[[s in str(k).upper() for s in PGX_SOURCES].index(True)]
                 for k in path if any(s in str(k).upper() for s in PGX_SOURCES)), "")


def _dip_label(d: dict) -> str:
    if d.get("label"):
        return str(d["label"])
    names = [(d.get(k) or {}).get("name") for k in ("allele1", "allele2")]
    return "/".join(n for n in names if n) or "–"


def _text(x) -> str:
    """Plain text from PharmCAT fields, which may be lists, dicts, or HTML (DPWG) with entities."""
    if isinstance(x, list):
        return "; ".join(_text(v) for v in x if v)
    if isinstance(x, dict):
        return _text(x.get("text") or x.get("name") or x.get("html") or "")
    t = html.unescape(re.sub(r"<[^>]+>", " ", str(x or "")))
    return re.sub(r"\s+", " ", t).strip().strip('"')


def load_pharmcat(out: Path) -> dict | None:
    """Genes and drug recommendations from PharmCAT's report JSON (or its phenotype JSON). The layout differs
    between PharmCAT releases, so records are found by their keys rather than by fixed paths."""
    d = out / "pharmcat"
    src = next((p for p in (d / "sample.report.json", d / "sample.phenotype.json") if p.exists()), None)
    if src is None:
        return None
    try:
        data = json.loads(src.read_text())
    except (OSError, ValueError) as e:
        print(f"Warning: could not read {src}: {e}")
        return None
    genes: dict[str, dict] = {}
    recs: dict[tuple, dict] = {}
    for path, o, up in _walk(data):
        if "geneSymbol" in o and ("sourceDiplotypes" in o or "recommendationDiplotypes" in o):
            dips = o.get("sourceDiplotypes") or o.get("recommendationDiplotypes") or []
            labels = list(dict.fromkeys(_dip_label(x) for x in dips))
            phen = list(dict.fromkeys(p for x in dips for p in (x.get("phenotypes") or []) if p))
            score = [str(x["activityScore"]) for x in dips if x.get("activityScore") not in (None, "", "n/a")]
            variants = [v for v in (o.get("variants") or []) if isinstance(v, dict)]
            called = (any(l.lower() not in NOT_CALLED and "unknown" not in l.lower() for l in labels)
                      and o.get("callSource") not in ("NONE",))
            g = genes.setdefault(o["geneSymbol"], {"gene": o["geneSymbol"], "called": False, "phen": {},
                                                    "diplotype": "–", "score": "", "pos": None})
            if called and not g["called"]:
                g.update(called=True, diplotype=" or ".join(labels), score=", ".join(score))
            if phen:
                g["phen"].setdefault(str(_source(path) or o.get("phenotypeSource") or ""), ", ".join(phen))
            if variants:
                have = sum(1 for v in variants if v.get("call") not in (None, "", "."))
                g["pos"] = (have, len(variants))
        rec = o.get("drugRecommendation")
        if isinstance(rec, (str, dict)) and _text(rec).strip():
            drug = ""
            if "drugs" in path:
                i = path.index("drugs")
                drug = next((str(k) for k in path[i + 1:i + 3]
                             if isinstance(k, str) and not _source((k,)) and "guideline" not in k.lower()), "")
            if not drug:
                drug = next((_text(a.get("name")) for a in reversed(up) if a.get("name")), "?")
            gene_names = sorted({str(x.get("gene") or x.get("geneSymbol")) for _, x, _ in _walk(o.get("genotypes", []))
                                 if x.get("gene") or x.get("geneSymbol")})
            pheno = sorted({p for _, x, _ in _walk(o.get("genotypes", [])) for p in (x.get("phenotypes") or [])})
            r = {"drug": drug, "source": _source(path), "genes": ", ".join(gene_names),
                 "phenotype": ", ".join(pheno), "class": _text(o.get("classification")),
                 "rec": _text(rec).strip(), "impl": _text(o.get("implications"))}
            recs.setdefault((r["drug"].lower(), r["source"], r["rec"]), r)
    stats = read_json(d / "pgx_input_stats.json")
    return {"genes": sorted(genes.values(), key=lambda g: (not g["called"], g["gene"])),
            "recs": list(recs.values()), "stats": stats, "src": src,
            "html": next((p for p in (d / "sample.report.html",) if p.exists()), None)}


def sec_pharmcat(out: Path, rep: Path) -> str:
    pc = load_pharmcat(out)
    if pc is None:
        if (out / "pharmcat" / "sample.pgx.grch38.vcf").exists():
            return section("pgx", "Drug response (PharmCAT)", "", card("PharmCAT did not finish",
                           f"<p>See {esc(out / 'pharmcat' / 'pharmcat.log')}.</p>"))
        return ""
    genes, recs, st = pc["genes"], pc["recs"], pc["stats"] or {}
    n_called = sum(g["called"] for g in genes)
    link = ""
    if pc["html"]:
        (rep / "pharmcat_full.html").write_bytes(pc["html"].read_bytes())
        link = ' <a href="pharmcat_full.html">PharmCAT\'s full report →</a>'
    tiles = [tile("Genes called", f"{n_called} of {len(genes)}", "the rest: no data or not callable from SNPs")]
    if st:
        tiles.append(tile("PharmCAT positions in your file", f"{st.get('written', 0):,} of {st.get('snv', 0):,}",
                          "SNVs; indels are not in the raw file"))
    actionable = [r for r in recs if pgx_actionable(r)]
    tiles.append(tile("Recommendations that change something", f"{len(actionable)}", f"of {len(recs)} drug–guideline pairs"))
    rows = []
    for g in genes:
        phen = "; ".join(f"{v} ({k})" if k and len(g["phen"]) > 1 else v for k, v in g["phen"].items()) or "–"
        pos = f"{g['pos'][0]} / {g['pos'][1]}" if g["pos"] else ""
        rows.append({"Gene": g["gene"], "Diplotype": g["diplotype"] if g["called"] else "no call",
                     "Phenotype": phen if g["called"] else "–", "Activity score": g["score"] or "",
                     "Positions with data": pos})
    gt = pd.DataFrame(rows)
    if not gt["Positions with data"].astype(bool).any():
        gt = gt.drop(columns="Positions with data")
    if not gt["Activity score"].astype(bool).any():
        gt = gt.drop(columns="Activity score")
    called = gt[gt["Diplotype"] != "no call"]
    body = f'<div class="tiles">{"".join(tiles)}</div>'
    body += card("Your pharmacogenes", table(called, "dense sortable") if len(called) else "<p>No gene could be called.</p>",
                 "Star-allele diplotypes and the phenotype CPIC/DPWG assign to them." + link)
    uncalled = gt[gt["Diplotype"] == "no call"]
    if len(uncalled):
        body += details(f"{len(uncalled)} genes without a call: " + ", ".join(uncalled["Gene"]),
                        "<p class=\"chart-note\">CYP2D6 needs copy-number and structural calls, HLA genes need typing; "
                        "others lack their key positions in the raw file.</p>")
    def rec_table(rs):
        rs = sorted(rs, key=lambda r: (STRENGTH.get(r["class"].lower().split()[0] if r["class"] else "", 9), r["drug"]))
        return table(pd.DataFrame([{"Drug": r["drug"], "Guideline": r["source"], "Genes": r["genes"],
                                    "Your phenotype": r["phenotype"], "Strength": r["class"],
                                    "Recommendation": r["rec"]} for r in rs]), "dense sortable")
    if actionable:
        body += card("Drugs where your genotype changes the advice", rec_table(actionable),
                     "From CPIC / DPWG guidelines as applied by PharmCAT. Talk to a doctor or pharmacist before acting on any of it.")
    routine = [r for r in recs if r not in actionable]
    if routine:
        body += details(f"Show {len(routine)} drugs with standard dosing for your genotype", rec_table(routine))
    lede = ("Imputed SNP data: a gene whose key positions are missing can come out as the reference allele (*1), so "
            "check “Positions with data” before trusting a normal result. Not a clinical test.")
    return section("pgx", "Drug response (PharmCAT)", esc(lede), body)
