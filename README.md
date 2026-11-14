# Personal DNA analysis (MyHeritage raw file + CRAM)

Inputs, all set in `configs/config.yaml`:
- `raw_file`: `MyHeritage_raw_dna_data.csv`, ~585k SNPs, GRCh37 forward strand, imputed from low-pass WGS,
  autosomes + X only (no Y / MT, so no haplogroups);
- `cram` + `reference_fasta`: the ~1.9× low-pass reads (Ultima) and their Broad b37 reference;
- `ref_dir`: downloaded references (`/mnt/d/My_DNA`, i.e. `D:\My_DNA` in Windows);
- `ancient_dir`: ancient BAMs/CRAMs for ancIBD (`/mnt/e/My_DNA_ancient`, external disk E:).

Everything is written to `out_dir` (`outputs/`). Environment:

```bash
mamba env create -f environment.yml && conda activate my_analysis
```

## Layout

| Folder | What it answers |
|---|---|
| `scripts/setup/` | downloads: 1000 Genomes, ClinVar, AADR, PharmCAT, ancIBD, ancient BAMs |
| `scripts/genome/` | your data: QC, VCFs, merge with 1000 Genomes, CRAM QC, GLIMPSE2 imputation |
| `scripts/health/` | ClinVar variants, polygenic scores |
| `scripts/drugs/` | pharmacogenomics (PharmCAT) |
| `scripts/traits/` | single-SNP traits, GWAS Catalog lookup, trait map and explorer |
| `scripts/ethnicity/` | present-day ancestry: PCA + ADMIXTURE, G25-style coordinates; ROH and relatives |
| `scripts/ancient/` | deep ancestry: AADR dataset, qpAdm, f3/f4 statistics, ancient PCA, Neanderthal, shared alleles |
| `scripts/matches/` | shared segments with individual ancient people: IBS scans with controls, ancIBD |
| `scripts/report/` | the HTML report built from everything above |

`scripts/common.py` (Python) and `scripts/env.sh` (shell: strict mode, config values as `$OUT`, `$REF`,
`$THREADS`…) are shared by all of them. Run every script from the project root; the blocks below are in
run order. Large downloads are commands for you to run, never run automatically.

## genome: your data

```bash
python scripts/genome/qc.py             # call rate, heterozygosity, inferred sex (no references needed)
bash scripts/setup/download_refs.sh     # 1000 Genomes (~15 GB, verified, resumable) + ClinVar; or setup/download_1kg.sh [chroms...]
bash scripts/genome/merge_1kg.sh        # you + 1000G -> outputs/merged.{pgen,pvar,psam} (runs genome/make_vcf.py)
bash scripts/genome/cram_qc.sh          # idxstats, coverage, chr1 depth histogram; then notebooks/*.ipynb
```

GLIMPSE2 imputation of the CRAM (needed by `matches/`). Chromosomes run in parallel as separate calls,
each resumable; THREADS overrides the config value:

```bash
THREADS=4 nohup bash scripts/genome/glimpse.sh outputs/glimpse/me.tsv me 1 4 7 10 13 16 19 22 > outputs/glimpse/me_a.log 2>&1 &
THREADS=4 nohup bash scripts/genome/glimpse.sh outputs/glimpse/me.tsv me 2 5 8 11 14 17 20   > outputs/glimpse/me_b.log 2>&1 &
THREADS=4 nohup bash scripts/genome/glimpse.sh outputs/glimpse/me.tsv me 3 6 9 12 15 18 21   > outputs/glimpse/me_c.log 2>&1 &
```
`CLEAN_SPLIT=1` deletes each chromosome's binary reference (~35 GB in all) after use. Start long runs from a
Windows Terminal tab and keep it open: WSL stops when the last window that started it closes.

## health and drugs

```bash
python scripts/health/clinvar.py        # ClinVar carriers (confirm anything clinically!)
python scripts/health/prs.py            # polygenic scores vs 1000G superpopulation
bash scripts/setup/download_pharmcat.sh # PharmCAT jar + GRCh38 positions (~50 MB) into ref_dir/pharmcat/
bash scripts/drugs/pharmcat.sh          # PharmCAT; Java: conda create -n java -c conda-forge --override-channels openjdk=21
```

## traits

```bash
python scripts/traits/snp_traits.py [--apoe]           # single-SNP traits from config `traits`
python scripts/traits/gwas_catalog.py                  # GWAS Catalog associations (needs `pip install requests`)
python scripts/traits/gwas_catalog.py --rsid rs12913832 rs4988235   # selected rsIDs; --refresh ignores outputs/gwas_cache/
python scripts/traits/plot_gwas_catalog.py [--top 20 --per-rsid 3]   # outputs/gwas_catalog_associations.png
python scripts/traits/gwas_trait_map.py [--per-category 5]           # trait categories + traits linked to alleles you carry
python scripts/traits/gwas_explorer.py                 # outputs/gwas_explorer.html: offline, filterable table
```

## ethnicity: present-day ancestry and family

```bash
bash scripts/ethnicity/ancestry.sh      # PCA + supervised ADMIXTURE + plot (ethnicity/plot_ancestry.py)
python scripts/ethnicity/g25_style.py   # G25-style coordinates (25 scaled PCs), distances, Vahaduo-style mixtures (~25 min; needs ancient/ data)
bash scripts/ethnicity/roh.sh           # runs of homozygosity
python scripts/ethnicity/relatives.py --other relative_raw.csv --name mum
```

## ancient: deep ancestry (AADR)

1000 Genomes has no Eastern Europeans apart from Finns, so these steps use the Allen Ancient DNA Resource
(Human Origins panel), which has modern Russian, Ukrainian and Volga-Uralic groups as well as ancient genomes.

```bash
bash scripts/setup/download_aadr.sh                 # AADR v66.p1 HO, ~4.1 GB, MD5-verified
python scripts/ancient/aadr_prepare.py check        # are the group labels in config `aadr.groups` real?
python scripts/ancient/aadr_prepare.py find Udmurt Yamnaya   # search labels to fix them
python scripts/ancient/aadr_prepare.py build        # extract groups + you (CRAM pass ~30+ min, reused afterwards)
python scripts/ancient/qpadm.py                     # ADMIXTOOLS qpAdm, all models in config `qpadm.analyses`, resumable
python scripts/ancient/qpadm_fast.py                # the same models in NumPy, in seconds
python scripts/ancient/qpadm_replicates.py pileup && python scripts/ancient/qpadm_replicates.py run -r 30   # repeat the random-read draw
python scripts/ancient/qpadm_east.py --replicates 10   # which population carries the Siberian-related share
python scripts/ancient/f3_ancient.py                # outgroup f3(Mbuti; you, X) for ~15k ancient individuals
python scripts/ancient/f4_uralic.py                 # f4(Early Slavs, X; Ukrainian, you) vs Russian/Belarusian/Ukrainian controls
python scripts/ancient/farmer_f4.py                 # f4(Mbuti, Anatolia_N; EHG/WHG/Yamnaya, test) + admixture f3
python scripts/ancient/pca_ancient.py               # West Eurasian PCA, ancient genomes and you projected (lsqproject-style)
python scripts/ancient/neanderthal.py               # Neanderthal ancestry (f4-ratio) with control individuals
python scripts/ancient/shared_alleles.py            # alleles rare today but common in ancient groups: rates vs controls
```

You enter qpAdm twice: `me_cram` (one random read per SNP from the CRAM, unaffected by imputation) and
`me_csv` (the imputed MyHeritage genotypes). Trust a model when it fits for both. `modern` asks whether
a Uralic/Volga source is needed on top of Slavic ones; `modern_wide` tries Baltic, Finnic, Volga-Turkic and
Central European groups (up to two at a time); `deep` gives WHG / EHG / Anatolia_N / Yamnaya / Nganasan
proportions; `proximal` uses closer-in-time Corded Ware and Early Slav sources. Run a subset with
`--analysis modern_wide proximal`; after adding groups to the config, rerun `ancient/aadr_prepare.py build`.
Prefer the simplest model that fits (p ≥ 0.01, all weights in [0, 1]).

## matches: shared segments with ancient individuals

```bash
python scripts/matches/ibs_ancient.py   # GEDmatch-style "matches" with famous genomes vs 1000G/AADR controls
python scripts/matches/ibs_all.py       # Ancestrify-style IBS scan: you (GLIMPSE-imputed) + 25 controls vs all 14,743 ancients
```

Real IBD (ancIBD): GLIMPSE2 imputes ~30 medieval Eastern European shotgun genomes as well, then ancIBD
looks for shared segments ≥ 8 cM.

```bash
python scripts/matches/ancibd.py select                # individuals from the AADR annotation (config `ancibd`)
bash scripts/setup/download_ancient_bams.sh manifest    # ENA file lists -> outputs/ancibd/manifest.tsv
bash scripts/setup/download_ancient_bams.sh fetch       # ~102 GB from ENA to ancient_dir, kept as ~45 GB CRAM; resumable
bash scripts/setup/download_ancibd.sh                   # ancIBD (own conda env) + its 1240k files
bash scripts/genome/glimpse.sh outputs/ancibd/glimpse_samples.tsv ancient
python scripts/matches/ancibd.py run && python scripts/matches/ancibd.py summary
```

## report

```bash
python scripts/report/build_report.py   # outputs/report/*.html + figures; opens Firefox (--no-open to skip)
```

One page per module, with tabs: Overview (headline cards) · Genome · Health · Drugs · Traits · Ethnicity ·
Ancient · Matches, plus detail pages (qpAdm, ancient individuals, map, PharmCAT's report, GWAS explorer).
Built from whatever outputs exist so far; rerun it after any step. The Ancestrify-style scan joins the Matches
page once `matches/ibs_all.py` has covered all 22 chromosomes. It profiles every matched ancient person from the AADR annotation (dating, site, sex,
haplogroups, data, paper) and flags any stretch where everyone's segments pile up (a false-match hotspot). Code: `report/core.py` (palette, HTML helpers,
page shell) and `report/sections/<module>.py`. The copy to keep is `D:\My_DNA\report`.

## Troubleshooting (WSL2)

- **`admixture` segfaults** right after `Size of G`: the 2018 static binary needs the legacy vsyscall page.
  Put `[wsl2]` / `kernelCommandLine = vsyscall=emulate` in `C:\Users\<you>\.wslconfig`, then `wsl --shutdown`.
- **PharmCAT input**: your genotypes are matched to PharmCAT's GRCh38 positions by rsID (no liftover, no
  Docker). The raw file has no indels and lacks some key SNPs (e.g. CYP2C19*2, CYP2C9*3, VKORC1), so those
  genes may come out as *1 by default; check "Positions with data" in the report. CYP2D6 is not callable.
- **Keep large downloads off `/mnt/d` until verified**: `setup/download_1kg.sh` stages in `~/1kg_staging` and
  checks every file with `bgzip -t` before and after moving it. WSL's own disk is a file on C:, which has
  little free space, so large files go to D: (or E: for the ancient BAMs).
- **GLIMPSE2_ligate "Three files overlapping"** near a centromere: `genome/glimpse.sh` cuts such chunks
  before ligation; rerun the chromosome and it ligates in seconds.

Caveats: rare variants from imputed/array data are often wrong; PRS percentiles depend on ancestry match;
GWAS associations are population-level findings, not personal medical predictions. The GWAS Catalog
script sends rsIDs only (not genotypes) to its public API. These analyses are not medical tests.
