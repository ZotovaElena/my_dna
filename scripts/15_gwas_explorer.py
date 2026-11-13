"""Build a self-contained, offline HTML explorer for all GWAS Catalog associations.

Reads gwas_trait_map.csv from 14_gwas_trait_map.py and writes gwas_explorer.html to
out_dir. Paper titles and abstracts are fetched from PubMed (only public PubMed IDs
are sent) and cached in <out_dir>/pubmed_cache.json; use --no-pubmed to skip.

The page embeds your genotypes, so keep it local. It loads nothing from the internet
when opened (only the PubMed/study/variant links you click open external sites).
"""

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

from common import config_arg, load_config

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_BATCH = 200
CONCLUSION_LABEL = re.compile(r"conclu|interpret|significance|summary", re.IGNORECASE)

FIELDS = [
    "rsid", "mapped_genes", "genotype", "effect_allele", "effect_allele_copies",
    "strand_ambiguous", "trait_label", "reported_trait", "category", "direction",
    "neg_log10_p", "pvalue_mantissa", "pvalue_exponent", "pvalue_description",
    "or_value", "beta", "ci_lower", "ci_upper", "risk_frequency",
    "accession_id", "pubmed_id", "first_author",
    "effect_size_tier", "effect_size_text", "for_you", "explanation",
]


def text_of(node) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def parse_pubmed(xml_bytes: bytes) -> dict[str, dict]:
    """Title, journal, year, abstract and conclusion for each article in an efetch reply."""
    papers = {}
    for article in ET.fromstring(xml_bytes).iter("PubmedArticle"):
        pmid = text_of(article.find("MedlineCitation/PMID"))
        info = article.find("MedlineCitation/Article")
        if not pmid or info is None:
            continue
        date = info.find("Journal/JournalIssue/PubDate")
        year = text_of(date.find("Year")) if date is not None else ""
        if not year and date is not None:
            year = text_of(date.find("MedlineDate"))[:4]
        sections = [(s.get("Label", ""), text_of(s))
                    for s in info.findall("Abstract/AbstractText")]
        abstract = "\n\n".join(f"{label}: {body}" if label else body for label, body in sections)
        conclusion = " ".join(body for label, body in sections if CONCLUSION_LABEL.search(label))
        conclusion_labeled = bool(conclusion)
        if not conclusion and sections:
            sentences = re.split(r"(?<=[.!?])\s+", sections[-1][1])
            conclusion = " ".join(sentences[-2:])
        papers[pmid] = {
            "title": text_of(info.find("ArticleTitle")),
            "journal": text_of(info.find("Journal/Title")),
            "year": year,
            "abstract": abstract,
            "conclusion": conclusion,
            "conclusion_labeled": conclusion_labeled,
        }
    return papers


def load_papers(pmids: list[str], cache_path: Path, offline: bool) -> dict[str, dict]:
    """PubMed records for the given IDs, fetching only those not already cached."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [p for p in pmids if p not in cache]
    if missing and not offline:
        session = requests.Session()
        for start in range(0, len(missing), PUBMED_BATCH):
            batch = missing[start:start + PUBMED_BATCH]
            try:
                response = session.post(
                    EFETCH_URL,
                    data={"db": "pubmed", "id": ",".join(batch), "retmode": "xml",
                          "tool": "my_dna_gwas_explorer"},
                    timeout=120,
                )
                response.raise_for_status()
                cache.update(parse_pubmed(response.content))
            except (requests.RequestException, ET.ParseError) as exc:
                print(f"WARNING: PubMed fetch failed ({exc}); continuing without "
                      f"{len(batch)} papers", file=sys.stderr)
            time.sleep(0.4)  # NCBI allows 3 requests/s without an API key
        cache_path.write_text(json.dumps(cache, ensure_ascii=False))
        print(f"PubMed: fetched {len([p for p in missing if p in cache])} new papers")
    return {p: cache[p] for p in pmids if p in cache}


def snp_context(cfg: dict) -> dict[str, dict]:
    """Curated per-SNP notes from the config's traits list."""
    return {rsid: {k: str(v) for k, v in info.items()}
            for rsid, info in (cfg.get("traits") or {}).items() if isinstance(info, dict)}


def embed(value) -> str:
    """JSON for a <script> block; '<' is escaped so text can never close the block."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GWAS Explorer</title>
<style>
:root {
  --surface: #fcfcfb; --raised: #ffffff; --ink: #0b0b0b; --ink-2: #4a4945; --muted: #6f6d68;
  --line: #e1e0d9; --hover: #f3f2ee; --accent: #2a78d6;
  --c0: #e6e5df; --c1: #3987e5; --c2: #184f95; --c0-ink: #4a4945; --c-ink: #ffffff;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #161614; --raised: #1f1f1c; --ink: #f2f1ec; --ink-2: #c9c7c0; --muted: #9b9991;
    --line: #34332f; --hover: #262622; --accent: #6da7ec;
    --c0: #383835; --c1: #3987e5; --c2: #86b6ef; --c0-ink: #c9c7c0; --c-ink: #0b0b0b;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1400px; margin: 0 auto; padding: 24px 20px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; font-weight: 600; }
.sub { color: var(--muted); margin: 0 0 20px; max-width: 75ch; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
.tile { background: var(--raised); border: 1px solid var(--line); border-radius: 10px;
  padding: 10px 14px; min-width: 150px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .k { color: var(--muted); font-size: 12px; }
.filters { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: end;
  padding: 12px 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
  position: sticky; top: 0; background: var(--surface); z-index: 2; }
.filters label { display: flex; flex-direction: column; gap: 3px; font-size: 12px; color: var(--muted); }
.filters input[type=search], .filters select, .filters input[type=number] {
  font: inherit; color: var(--ink); background: var(--raised); border: 1px solid var(--line);
  border-radius: 6px; padding: 5px 8px; min-width: 0; }
.filters input[type=search] { width: 240px; max-width: 100%; }
.filters select { width: 220px; max-width: 100%; }
.filters .check { flex-direction: row; align-items: center; gap: 6px; color: var(--ink); font-size: 13px;
  padding-bottom: 6px; }
button { font: inherit; color: var(--ink); background: var(--raised); border: 1px solid var(--line);
  border-radius: 6px; padding: 5px 10px; cursor: pointer; }
button:hover { background: var(--hover); }
.count { color: var(--muted); font-size: 13px; margin: 10px 0; }
.tablebox { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: 12px; color: var(--muted); font-weight: 600; cursor: pointer; white-space: nowrap;
  user-select: none; }
th[aria-sort=ascending]::after { content: " \25B2"; }
th[aria-sort=descending]::after { content: " \25BC"; }
tbody tr.main { cursor: pointer; }
tbody tr.main:hover { background: var(--hover); }
td.num { text-align: right; }
.snp { font-weight: 600; }
.gene, .small { color: var(--muted); font-size: 12px; }
.badge { display: inline-block; border-radius: 999px; padding: 1px 8px; font-size: 12px; white-space: nowrap; }
.b0 { background: var(--c0); color: var(--c0-ink); }
.b1 { background: var(--c1); color: var(--c-ink); }
.b2 { background: var(--c2); color: var(--c-ink); }
.bu { border: 1px dashed var(--line); color: var(--muted); }
.dir { white-space: nowrap; color: var(--ink-2); }
tr.detail td { background: var(--raised); color: var(--ink-2); font-size: 13px; }
tr.detail dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 4px 0; }
tr.detail dt { color: var(--muted); }
tr.detail dd { margin: 0; }
a { color: var(--accent); }
.guide { background: var(--raised); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 16px; margin-bottom: 20px; max-width: 95ch; color: var(--ink-2); font-size: 13px; }
.guide summary { cursor: pointer; color: var(--ink); font-weight: 600; font-size: 14px; }
.guide ul { margin: 8px 0 0; padding-left: 20px; }
.guide li { margin: 4px 0; }
.verdict { display: inline-block; white-space: nowrap; font-weight: 600; }
.size { display: block; color: var(--muted); font-size: 12px; font-weight: 400; }
.explain { color: var(--ink); font-size: 14px; margin: 2px 0 10px; max-width: 95ch; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px 28px; }
.cols h4 { margin: 4px 0 6px; font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
  color: var(--muted); font-weight: 600; }
.paper-title { color: var(--ink); font-weight: 600; }
.paper p { margin: 6px 0; }
.paper details summary { cursor: pointer; color: var(--accent); font-size: 12px; }
.paper .abstract { white-space: pre-line; margin-top: 6px; }
h2 { font-size: 17px; margin: 0 0 4px; font-weight: 600; }
.summary { margin-bottom: 24px; }
.summary .sub { margin-bottom: 12px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
.card { background: var(--raised); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.card h3 { margin: 0; font-size: 14px; font-weight: 600; }
.card .meta { color: var(--muted); font-size: 12px; margin: 2px 0 8px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font-size: 12px; padding: 2px 9px; border-radius: 999px; border: 1px solid var(--line);
  background: var(--surface); color: var(--ink-2); cursor: pointer; text-align: left; }
.chip:hover, .chip[aria-pressed=true] { border-color: var(--accent); color: var(--ink); }
.chip .mark { color: var(--accent); margin-right: 3px; }
.more { font-size: 12px; border: none; background: none; color: var(--accent); padding: 2px 4px; }
.active-trait { display: inline-flex; gap: 6px; align-items: center; margin-left: 10px; }
.foot { color: var(--muted); font-size: 12px; margin-top: 24px; max-width: 80ch; }
</style>
</head>
<body>
<div class="wrap">
  <h1>GWAS Catalog explorer</h1>
  <p class="sub">Every GWAS Catalog association for your selected SNPs, with your genotype.
  These are population-level statistical associations, usually with small effects per allele.
  They are not personal risk predictions or medical results.</p>

  <div class="tiles" id="tiles"></div>

  <details class="guide">
    <summary>How to read “For you”</summary>
    <ul>
      <li><b>▲ toward higher / ▼ toward lower</b>: what your genotype points to for that trait
        (a value such as cholesterol, or the risk of a disease), compared with people with the other
        genotype in the study. It describes a direction, not a diagnosis.</li>
      <li><b>Effect size</b>: how big the difference is per copy of the allele. Most results are
        <i>very small</i> or <i>small</i>: odds ratios under 1.5, or a few hundredths of a standard
        deviation. Alone, they barely change anyone’s chances.</li>
      <li><b>? unclear</b>: the SNP isn’t in your file, the study didn’t say which allele has the
        effect or in which direction, or the strand can’t be resolved.</li>
      <li>There is no “no risk” verdict. A GWAS only compares genotypes, and every trait also
        depends on many other genes, lifestyle and environment. One SNP can’t rule a risk in or out.</li>
      <li>A very low p-value means the association is statistically certain in large populations.
        It does <i>not</i> mean the effect is large.</li>
    </ul>
  </details>

  <section class="summary">
    <h2>What traits the catalog links to your SNPs</h2>
    <p class="sub" id="summary-sub"></p>
    <div class="cards" id="cards"></div>
  </section>

  <div class="filters" role="search">
    <label>Search trait, gene or study
      <input type="search" id="q" placeholder="e.g. diabetes, HERC2, Verma"></label>
    <label>SNP <select id="rsid"><option value="">All SNPs</option></select></label>
    <label>Category <select id="category"><option value="">All categories</option></select></label>
    <label>For you <select id="foryou">
      <option value="">Any</option><option value="toward higher">▲ Toward higher</option>
      <option value="toward lower">▼ Toward lower</option><option value="unclear">? Unclear</option></select></label>
    <label>Effect size <select id="size">
      <option value="">Any</option><option value="large">Large</option>
      <option value="moderate+">Moderate or large</option><option value="known">Any known size</option></select></label>
    <label>Min −log10(p) <input type="number" id="minp" value="7.3" step="0.1" min="0" style="width:90px"></label>
    <label class="check"><input type="checkbox" id="carried"> Only alleles I carry</label>
    <label class="check"><input type="checkbox" id="dedupe" checked> Best result per SNP × trait</label>
    <button id="reset" type="button">Reset</button>
  </div>

  <div class="count" id="count" aria-live="polite"></div>
  <div class="tablebox">
    <table>
      <thead><tr>
        <th data-key="rsid">SNP</th>
        <th data-key="genotype">Your genotype</th>
        <th data-key="effect_allele">Effect allele</th>
        <th data-key="effect_allele_copies">Your copies</th>
        <th data-key="trait_label">Trait</th>
        <th data-key="category">Category</th>
        <th data-key="for_you">For you</th>
        <th data-key="neg_log10_p" class="num" aria-sort="descending">−log10(p)</th>
        <th data-key="effect" class="num">OR / beta</th>
        <th data-key="first_author">Study</th>
      </tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
  <p class="foot">Click a row for details. “Your copies” counts the reported effect allele in your
  genotype. It is left blank for A/T and C/G SNPs, where the strand can’t be resolved, and where the
  catalog gives no allele. Direction comes from the odds ratio, or from “increase/decrease” in the
  reported beta. Data: NHGRI-EBI GWAS Catalog. Links open external sites.</p>
</div>

<script id="data" type="application/json">__DATA__</script>
<script id="papers" type="application/json">__PAPERS__</script>
<script id="snps" type="application/json">__SNPS__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const PAPERS = JSON.parse(document.getElementById("papers").textContent);
const SNPS = JSON.parse(document.getElementById("snps").textContent);
const GW = 7.30103;
const SIZE_RANK = { "very small": 1, small: 2, moderate: 3, large: 4 };
const $ = (id) => document.getElementById(id);
const els = { q: $("q"), rsid: $("rsid"), category: $("category"), foryou: $("foryou"),
  size: $("size"), minp: $("minp"), carried: $("carried"), dedupe: $("dedupe") };
let sortKey = "neg_log10_p", sortDir = -1, openRow = null, traitFilter = null;

DATA.forEach((d, i) => {
  d._i = i;
  d.copies = d.strand_ambiguous === true || d.strand_ambiguous === "True" ? null : d.effect_allele_copies;
  d.effect = d.or_value != null ? d.or_value : null;
  const paper = PAPERS[d.pubmed_id] || {};
  d._search = [d.trait_label, d.reported_trait, d.mapped_genes, d.rsid, d.first_author,
    d.accession_id, d.category, paper.title].filter(Boolean).join(" ").toLowerCase();
  d._size = SIZE_RANK[d.effect_size_tier] || 0;
});

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else n.setAttribute(k, v);
  }
  kids.forEach((k) => n.append(k));
  return n;
}
function fill(select, values) {
  values.forEach((v) => select.append(el("option", { value: v, text: v })));
}
const uniq = (key) => [...new Set(DATA.map((d) => d[key]).filter(Boolean))].sort();
const snpLabel = {};
DATA.forEach((d) => { snpLabel[d.rsid] = `${d.rsid}${d.mapped_genes ? " (" + d.mapped_genes + ")" : ""}`; });
uniq("rsid").forEach((r) => els.rsid.append(el("option", { value: r, text: snpLabel[r] })));
fill(els.category, uniq("category"));

function tiles() {
  const gw = DATA.filter((d) => d.neg_log10_p >= GW);
  const carried = gw.filter((d) => d.copies >= 1);
  const t = (rows) => new Set(rows.map((d) => d.rsid + "|" + d.trait_label)).size;
  const items = [
    [DATA.length.toLocaleString(), "associations"],
    [new Set(DATA.map((d) => d.trait_label)).size, "distinct traits"],
    [new Set(DATA.map((d) => d.rsid)).size, "SNPs"],
    [t(carried), "SNP × trait pairs where you carry the allele (genome-wide)"],
  ];
  items.forEach(([v, k]) => $("tiles").append(el("div", { class: "tile" },
    el("div", { class: "v", text: v }), el("div", { class: "k", text: k }))));
}

const CHIPS_SHOWN = 10;
function summary() {
  const byCat = new Map();
  DATA.forEach((d) => {
    const cat = d.category || "Uncategorised";
    if (!byCat.has(cat)) byCat.set(cat, new Map());
    const traits = byCat.get(cat);
    if (!traits.has(d.trait_label)) traits.set(d.trait_label, { rows: [], snps: new Set(), best: -1 });
    const t = traits.get(d.trait_label);
    t.rows.push(d); t.snps.add(d.rsid); t.best = Math.max(t.best, d.neg_log10_p ?? -1);
  });
  const cats = [...byCat.entries()].sort((a, b) => b[1].size - a[1].size);
  const nTraits = cats.reduce((n, [, t]) => n + t.size, 0);
  $("summary-sub").textContent = `${nTraits} distinct traits in ${cats.length} GWAS Catalog categories, ` +
    "largest first. Chips are ordered by strongest p-value. ▲/▼ shows where your known results all point " +
    "the same way, ↕ shows mixed results. Click a chip to list its associations.";

  cats.forEach(([cat, traits]) => {
    const snps = new Set([...traits.values()].flatMap((t) => [...t.snps]));
    const list = [...traits.entries()].sort((a, b) => b[1].best - a[1].best);
    const chips = el("div", { class: "chips" });
    const card = el("div", { class: "card" },
      el("h3", { text: cat }),
      el("div", { class: "meta", text: `${traits.size} trait${traits.size > 1 ? "s" : ""} · ` +
        `${snps.size} SNP${snps.size > 1 ? "s" : ""} · ${[...snps].map((s) => geneOf(s)).join(", ")}` }),
      chips);
    list.forEach(([name, t], i) => {
      const verdicts = new Set(t.rows.map((r) => r.for_you).filter((v) => v && v !== "unclear"));
      const mark = verdicts.size === 0 ? "" : verdicts.size > 1 ? "↕"
        : verdicts.has("toward higher") ? "▲" : "▼";
      const chip = el("button", { type: "button", class: "chip", "aria-pressed": "false",
        title: `${t.rows.length} association(s) · ${[...t.snps].join(", ")}` });
      if (mark) chip.append(el("span", { class: "mark", text: mark }));
      chip.append(`${name} (${[...t.snps].join(", ")})`);
      chip.addEventListener("click", () => setTrait(name));
      chip.dataset.trait = name;
      if (i >= CHIPS_SHOWN) chip.hidden = true;
      chips.append(chip);
    });
    if (list.length > CHIPS_SHOWN) {
      const more = el("button", { type: "button", class: "more", text: `+ ${list.length - CHIPS_SHOWN} more` });
      more.addEventListener("click", () => {
        const expand = more.textContent.startsWith("+");
        chips.querySelectorAll(".chip").forEach((c, i) => { if (i >= CHIPS_SHOWN) c.hidden = !expand; });
        more.textContent = expand ? "Show fewer" : `+ ${list.length - CHIPS_SHOWN} more`;
      });
      chips.append(more);
    }
    $("cards").append(card);
  });
}
function geneOf(rsid) {
  const d = DATA.find((x) => x.rsid === rsid && x.mapped_genes);
  return d ? d.mapped_genes.split(";")[0].trim() : rsid;
}
function setTrait(name) {
  traitFilter = traitFilter === name ? null : name;
  if (traitFilter) {  // show every result for the trait, not only the current filter
    els.minp.value = "0"; els.carried.checked = false;
  }
  document.querySelectorAll(".chip").forEach((c) =>
    c.setAttribute("aria-pressed", String(c.dataset.trait === traitFilter)));
  openRow = null; render();
  if (traitFilter) document.querySelector(".filters").scrollIntoView({ behavior: "smooth" });
}

function filtered() {
  const q = els.q.value.trim().toLowerCase();
  const minp = parseFloat(els.minp.value);
  let rows = DATA.filter((d) =>
    (!traitFilter || d.trait_label === traitFilter) &&
    (!q || d._search.includes(q)) &&
    (!els.rsid.value || d.rsid === els.rsid.value) &&
    (!els.category.value || d.category === els.category.value) &&
    (!els.foryou.value || d.for_you === els.foryou.value) &&
    (!els.size.value || (els.size.value === "large" ? d._size === 4
      : els.size.value === "moderate+" ? d._size >= 3 : d._size >= 1)) &&
    (isNaN(minp) || (d.neg_log10_p != null && d.neg_log10_p >= minp)) &&
    (!els.carried.checked || d.copies >= 1));
  if (els.dedupe.checked) {
    const best = new Map();
    rows.forEach((d) => {
      const k = d.rsid + "|" + d.trait_label, cur = best.get(k);
      if (!cur || (d.neg_log10_p ?? -1) > (cur.neg_log10_p ?? -1)) best.set(k, d);
    });
    rows = [...best.values()];
  }
  return rows.sort((a, b) => {
    const x = a[sortKey], y = b[sortKey];
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * sortDir;
  });
}

function fmtP(d) {
  if (d.pvalue_mantissa != null && d.pvalue_exponent != null)
    return `${(+d.pvalue_mantissa).toPrecision(2)}×10${sup(d.pvalue_exponent)}`;
  return d.neg_log10_p != null ? `10${sup(-Math.round(d.neg_log10_p))}` : "—";
}
function sup(n) {
  const m = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  return String(Math.round(n)).split("").map((c) => m[c] ?? c).join("");
}
function copiesBadge(d) {
  if (d.copies == null) return el("span", { class: "badge bu", text: "unknown" });
  const c = Math.round(d.copies);
  return el("span", { class: `badge b${c}`, text: `${c} cop${c === 1 ? "y" : "ies"}` });
}
function verdict(d) {
  const map = { "toward higher": "▲ toward higher", "toward lower": "▼ toward lower" };
  const v = el("span", { class: "verdict", text: map[d.for_you] || "? unclear" });
  if (d.for_you !== "unclear" && d.effect_size_tier && d.effect_size_tier !== "unknown")
    v.append(el("span", { class: "size", text: `${d.effect_size_tier} effect` }));
  return v;
}
function effectText(d) {
  if (d.or_value != null) return `OR ${(+d.or_value).toFixed(2)}`;
  return d.beta ? String(d.beta) : "—";
}
function link(href, text) {
  return el("a", { href, text, target: "_blank", rel: "noopener noreferrer" });
}

function paperBlock(d) {
  const p = PAPERS[d.pubmed_id];
  const box = el("div", { class: "paper" }, el("h4", { text: "The study" }));
  if (!p) {
    box.append(el("p", { text: d.pubmed_id ? "No PubMed summary cached for this paper."
      : "No PubMed ID recorded." }));
    return box;
  }
  box.append(el("p", {},
    link(`https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(d.pubmed_id)}/`, p.title || `PubMed ${d.pubmed_id}`),
    el("span", { class: "small", text: ` ${d.first_author || ""}${p.journal ? ", " + p.journal : ""}${p.year ? " (" + p.year + ")" : ""}` })));
  if (p.conclusion) box.append(el("p", {},
    el("b", { text: p.conclusion_labeled ? "Authors’ conclusion: " : "End of abstract: " }), p.conclusion));
  if (p.abstract) {
    box.append(el("details", {}, el("summary", { text: "Full abstract" }),
      el("div", { class: "abstract", text: p.abstract })));
  }
  return box;
}

function detailRow(d) {
  const cell = el("td", { colspan: "10" });
  cell.append(el("p", { class: "explain", text: d.explanation || "" }));
  const snp = SNPS[d.rsid];
  if (snp) {
    cell.append(el("p", { class: "small" }, el("b", { text: `About ${d.rsid}: ` }),
      `${snp.trait || ""}${snp.effect_allele ? `. Best-known effect allele: ${snp.effect_allele}` : ""}` +
      `${snp.effect ? ` (${snp.effect})` : ""}. This result may concern a different trait.`));
  }
  const dl = el("dl");
  const add = (k, v) => { if (v != null && v !== "") dl.append(el("dt", { text: k }), el("dd", {}, v)); };
  add("Allele’s effect", { higher: "▲ higher", lower: "▼ lower" }[d.direction] || "not reported");
  add("Reported trait", d.reported_trait);
  add("p-value", fmtP(d) + (d.pvalue_description ? ` (${d.pvalue_description})` : ""));
  add("Effect", effectText(d) + (d.ci_lower != null ? `, 95% CI ${d.ci_lower}–${d.ci_upper}` : ""));
  add("Effect-allele frequency", d.risk_frequency);
  add("Mapped genes", d.mapped_genes);
  if (d.strand_ambiguous === true || d.strand_ambiguous === "True")
    add("Note", "A/T or C/G SNP: catalog allele strand is ambiguous, so copies aren’t counted.");
  const links = el("span");
  if (d.pubmed_id) links.append(link(`https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(d.pubmed_id)}/`, `PubMed ${d.pubmed_id}`), "  ·  ");
  if (d.accession_id) links.append(link(`https://www.ebi.ac.uk/gwas/studies/${encodeURIComponent(d.accession_id)}`, d.accession_id), "  ·  ");
  links.append(link(`https://www.ebi.ac.uk/gwas/variants/${encodeURIComponent(d.rsid)}`, `${d.rsid} in GWAS Catalog`));
  add("Links", links);
  const facts = el("div", {}, el("h4", { text: "Details" }), dl);
  cell.append(el("div", { class: "cols" }, facts, paperBlock(d)));
  return el("tr", { class: "detail" }, cell);
}

function render() {
  const rows = filtered();
  const body = $("rows");
  body.replaceChildren();
  const frag = document.createDocumentFragment();
  rows.forEach((d) => {
    const tr = el("tr", { class: "main", tabindex: "0", "aria-expanded": String(openRow === d._i) },
      el("td", {}, el("div", { class: "snp", text: d.rsid }), el("div", { class: "gene", text: d.mapped_genes || "" })),
      el("td", { text: d.genotype || "" }),
      el("td", { text: d.effect_allele || "?" }),
      el("td", {}, copiesBadge(d)),
      el("td", { text: d.trait_label || "" }),
      el("td", { class: "small", text: d.category || "" }),
      el("td", {}, verdict(d)),
      el("td", { class: "num", text: d.neg_log10_p != null ? d.neg_log10_p.toFixed(1) : "—" }),
      el("td", { class: "num small", text: effectText(d) }),
      el("td", { class: "small", text: d.first_author || "" }));
    const toggle = () => { openRow = openRow === d._i ? null : d._i; render(); };
    tr.addEventListener("click", toggle);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
    frag.append(tr);
    if (openRow === d._i) frag.append(detailRow(d));
  });
  body.append(frag);
  const count = $("count");
  count.replaceChildren(`Showing ${rows.length.toLocaleString()} of ${DATA.length.toLocaleString()} associations`);
  if (traitFilter) {
    const clear = el("button", { type: "button", text: "✕ clear" });
    clear.addEventListener("click", () => setTrait(traitFilter));
    count.append(el("span", { class: "active-trait" }, `for trait “${traitFilter}”`, clear));
  }
}

document.querySelectorAll("th[data-key]").forEach((th) => th.addEventListener("click", () => {
  const key = th.dataset.key;
  sortDir = key === sortKey ? -sortDir : (key === "neg_log10_p" || key === "effect" ? -1 : 1);
  sortKey = key;
  document.querySelectorAll("th").forEach((h) => h.removeAttribute("aria-sort"));
  th.setAttribute("aria-sort", sortDir === 1 ? "ascending" : "descending");
  render();
}));
Object.values(els).forEach((e) => e.addEventListener("input", () => { openRow = null; render(); }));
$("reset").addEventListener("click", () => {
  els.q.value = ""; els.rsid.value = ""; els.category.value = ""; els.foryou.value = "";
  els.size.value = ""; traitFilter = null;
  document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
  els.minp.value = "7.3"; els.carried.checked = false; els.dedupe.checked = true;
  openRow = null; render();
});
tiles();
summary();
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = config_arg(__doc__)
    parser.add_argument("--input", type=Path, help="trait map CSV (default: outputs/gwas_trait_map.csv)")
    parser.add_argument("--output", type=Path, help="HTML path (default: outputs/gwas_explorer.html)")
    parser.add_argument("--no-pubmed", action="store_true",
                        help="don't contact PubMed; use only papers already cached")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = args.input or cfg["out_dir"] / "gwas_trait_map.csv"
    output_path = args.output or cfg["out_dir"] / "gwas_explorer.html"
    if not input_path.exists():
        parser.error(f"{input_path} not found; run 14_gwas_trait_map.py first")

    report = pd.read_csv(input_path)
    missing = set(FIELDS) - set(report.columns)
    if missing:
        raise ValueError(f"{input_path} is missing columns {sorted(missing)}; re-run scripts 12 and 14")

    report["pubmed_id"] = (pd.to_numeric(report["pubmed_id"], errors="coerce")
                           .astype("Int64").astype(str).replace("<NA>", ""))
    records = json.loads(report[FIELDS].to_json(orient="records"))
    pmids = sorted({r["pubmed_id"] for r in records if r["pubmed_id"]})
    papers = load_papers(pmids, cfg["out_dir"] / "pubmed_cache.json", args.no_pubmed)

    html = (TEMPLATE.replace("__DATA__", embed(records))
            .replace("__PAPERS__", embed(papers))
            .replace("__SNPS__", embed(snp_context(cfg))))
    output_path.write_text(html, encoding="utf-8")
    print(f"Explorer with {len(records)} associations and {len(papers)}/{len(pmids)} "
          f"paper summaries saved to {output_path}")


if __name__ == "__main__":
    main()
