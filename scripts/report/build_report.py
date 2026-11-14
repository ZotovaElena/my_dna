"""HTML report from the pipeline outputs, one page per thematic module (as in scripts/).

Writes to out_dir/report/:
  index.html       overview: one card per module with its headline results
  genome.html      raw-file QC, sequencing depth                  (scripts/genome/)
  health.html      ClinVar, polygenic scores                       (scripts/health/)
  drugs.html       PharmCAT drug recommendations                   (scripts/drugs/)
  traits.html      single-SNP traits, GWAS Catalog associations    (scripts/traits/)
  ethnicity.html   ADMIXTURE + PCA, G25-style coordinates, ROH     (scripts/ethnicity/)
  ancient.html     qpAdm, closest ancients (f3), f4, farmer and Neanderthal ancestry, shared variants (scripts/ancient/)
  matches.html     IBS segments with ancient individuals           (scripts/matches/)
  qpadm.html, ancient_individuals.html, map.html, pharmcat_full.html, gwas_explorer.html: detail pages
  figures/         the main graphs as PNG (for sharing / printing)

Code: report/core.py (palette, HTML helpers, page shell) and report/sections/<module>.py. Each section is skipped
when its inputs don't exist yet, so this can run at any stage of the pipeline. The pages are self-contained
(the map loads its tiles online), follow the system light/dark setting, and show details on hover.
"""

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/: common.py and the module folders
from common import config_arg, load_config
from report import core
from report.core import MODULES, card, page, read_json, section
from report.sections.genome import fig_depth, load_coverage, sec_cram, sec_qc
from report.sections.health import sec_clinvar, sec_prs
from report.sections.drugs import sec_pharmcat
from report.sections.traits import load_gwas, sec_gwas, sec_snp_traits
from report.sections.ethnicity import fig_admixture, fig_roh, load_pca, load_roh, sec_ancestry, sec_g25, sec_roh
from report.sections.qpadm import fig_qpadm_models, fig_qpadm_p, load_qpadm, qpadm_page, qpadm_results_path, sec_deep_teaser
from report.sections.ancient import sec_f4, sec_farmer, sec_neanderthal, sec_shared
from report.sections.f3 import f3_page, load_f3, map_page, sec_f3_teaser
from report.sections.matches import sec_ibs, sec_ibs_all
from report.sections.overview import overview

FIREFOX = Path("/mnt/c/Program Files/Mozilla Firefox/firefox.exe")
SUBTITLES = {
    "genome": "Your raw data: the MyHeritage SNP file and the ~1.9× sequencing reads",
    "health": "ClinVar variants and polygenic scores. Not a medical test",
    "drugs": "Pharmacogenomics with PharmCAT and CPIC / DPWG guidelines",
    "traits": "Single-SNP traits and GWAS Catalog associations",
    "ethnicity": "Present-day ancestry: 1000 Genomes, AADR Human Origins populations",
    "ancient": "Your ancestry in terms of ancient populations (AADR v66)",
    "matches": "Shared DNA segments with individual ancient people",
}


def ibs_all_ready(out: Path) -> bool:
    """The Ancestrify-style scan is shown only once it covers all 22 chromosomes (GLIMPSE finished)."""
    st = read_json(out / "ibs_all" / "settings.json")
    return bool(st) and len(st.get("chromosomes", [])) == 22 and not st.get("limit")


def main() -> None:
    plt.switch_backend("Agg")
    parser = config_arg(__doc__)
    parser.add_argument("--no-open", action="store_true", help="don't open the report in Firefox afterwards")
    parser.add_argument("--qpadm-results", help="file in out_dir/qpadm/ (default: qpadm_fast_results.csv if it "
                                                "exists, else qpadm_results.csv)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = cfg["out_dir"]
    rep = out / "report"
    figs = rep / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    qc = read_json(out / "qc.json")
    anc = read_json(out / "ancestry.json")
    pcs = load_pca(cfg)
    roh = load_roh(out)
    cov = load_coverage(out)
    qp_path = qpadm_results_path(out, args.qpadm_results)
    qp = load_qpadm(qp_path)
    f3 = load_f3(out)
    gw, carried = load_gwas(out)

    # Module pages: (section id, nav label, html); a page with no sections is not written
    pending = section("pending", "Coming next", "", card(
        "Real IBD detection (ancIBD)",
        "<p>The scans above count runs of SNPs that don't contradict sharing (IBS). ancIBD looks for segments "
        "inherited from a common ancestor (IBD), modelling the genotype uncertainty of imputed ancient genomes; it "
        "needs about 30 medieval Eastern European genomes imputed like yours (setup/download_ancient_bams.sh, "
        "genome/glimpse.sh, then <code>matches/ancibd.py</code>). Its results will appear here.</p>"))
    pages = {
        "genome": [("qc", "At a glance", sec_qc(qc, cov)), ("sequencing", "Data quality", sec_cram(cov, qc))],
        "health": [("clinvar", "ClinVar", sec_clinvar(out)), ("prs", "Polygenic scores", sec_prs(out))],
        "drugs": [("pgx", "PharmCAT", sec_pharmcat(out, rep))],
        "traits": [("snps", "Trait SNPs", sec_snp_traits(out, gw)),
                   ("gwas", "GWAS Catalog", sec_gwas(out, rep, gw, carried))],
        "ethnicity": [("ancestry", "ADMIXTURE & PCA", sec_ancestry(cfg, anc, pcs)), ("g25", "G25-style", sec_g25(out, rep)),
                      ("roh", "Runs of homozygosity", sec_roh(roh))],
        "ancient": [("deep", "Deep ancestry", sec_deep_teaser(cfg, qp)), ("f3", "Closest ancients", sec_f3_teaser(f3)),
                    ("farmer", "Farmer ancestry", sec_farmer(cfg, out)), ("f4", "Uralic signal", sec_f4(out)),
                    ("neanderthal", "Neanderthal", sec_neanderthal(out)), ("shared", "Shared variants", sec_shared(out))],
        "matches": [("ibs", "GEDmatch-style scan", sec_ibs(out))]
                   + ([("ibs_all", "Ancestrify-style scan", sec_ibs_all(out, cfg))] if ibs_all_ready(out) else [])
                   + [("pending", "Coming next", pending)],
    }
    pages = {m: [p for p in parts if p[2]] for m, parts in pages.items()}
    pages = {m: parts for m, parts in pages.items() if any(i != "pending" for i, _, _ in parts)}
    core.TABS = ["index", *pages]
    titles = {m: t for m, t in MODULES}
    extra = {
        "drugs": [("pharmcat_full.html", "PharmCAT's full report")] if (rep / "pharmcat_full.html").exists() else [],
        "traits": [("gwas_explorer.html", "GWAS explorer")] if (rep / "gwas_explorer.html").exists() else [],
        "ancient": [(f, t) for f, t, ok in [("qpadm.html", "qpAdm details", qp is not None),
                                             ("ancient_individuals.html", "Ancient individuals", f3 is not None),
                                             ("map.html", "Map", f3 is not None)] if ok],
    }
    written = []
    for m, parts in pages.items():
        title = "Ancient ancestry" if m == "ancient" else titles[m]
        html_ = page(title, SUBTITLES[m], [(i, t) for i, t, _ in parts], "".join(h for _, _, h in parts),
                     active=m, extra=extra.get(m))
        (rep / f"{m}.html").write_text(html_)
        written.append(rep / f"{m}.html")

    sub = f"Sample “{cfg['sample_id']}”"
    if qc:
        sub += f" · {qc['header'].get('method', '')} · {qc['header'].get('reference', '')}"
    (rep / "index.html").write_text(page("My DNA report", sub, [], overview(out, cfg, qc, anc, roh, cov, qp, f3,
                                                                          {m: titles[m] for m in pages}),
                                         active="index"))
    written.insert(0, rep / "index.html")

    # Detail pages
    if qp is not None:
        (rep / "qpadm.html").write_text(qpadm_page(cfg, qp, qp_path))
        written.append(rep / "qpadm.html")
    if f3 is not None:
        (rep / "ancient_individuals.html").write_text(f3_page(f3))
        (rep / "map.html").write_text(map_page(f3))
        written += [rep / "ancient_individuals.html", rep / "map.html"]

    # PNG figures
    if qp is not None:
        for analysis in dict.fromkeys(qp["analysis"]):
            for target in dict.fromkeys(qp["target"]):
                path = figs / f"qpadm_{analysis}_{target}.png"
                if fig_qpadm_models(cfg, qp, analysis, target, path):
                    written.append(path)
            path = figs / f"qpadm_{analysis}_pvalues.png"
            fig_qpadm_p(qp, analysis, cfg["qpadm"]["p_min"], path)
            written.append(path)
    if anc:
        fig_admixture(anc, figs / "admixture.png")
        written.append(figs / "admixture.png")
    if roh is not None:
        fig_roh(roh, figs / "roh.png")
        written.append(figs / "roh.png")
    if cov and "hist" in cov:
        fig_depth(cov, figs / "depth_chr1.png")
        written.append(figs / "depth_chr1.png")

    print("Written:")
    for p in written:
        print(f"  {p}")
    index_win = subprocess.run(["wslpath", "-w", str(rep / "index.html")], capture_output=True, text=True).stdout.strip()
    if FIREFOX.exists() and not args.no_open and index_win:
        subprocess.Popen([str(FIREFOX), index_win], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        print(f"\nOpened in Firefox: {index_win}")
    else:
        print(f"\nOpen in Firefox: \"{FIREFOX}\" \"$(wslpath -w {rep / 'index.html'})\"")


if __name__ == "__main__":
    main()
