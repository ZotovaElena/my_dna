# Personal DNA analysis (MyHeritage raw file + CRAM)

Inputs, all set in `configs/config.yaml`:
- `raw_file`: `MyHeritage_raw_dna_data.csv`, ~585k SNPs, GRCh37 forward strand, imputed from low-pass WGS,
  autosomes + X only (no Y / MT, so no haplogroups);
- `cram` + `reference_fasta`: the ~1.9× low-pass reads (Ultima) and their Broad b37 reference;
- `ref_dir`: downloaded references (`/mnt/d/My_DNA`, i.e. `D:\My_DNA` in Windows).

Everything is written to `out_dir` (`outputs/`).

```bash
mamba env create -f environment.yml && conda activate my_analysis

# No references needed
python scripts/01_qc.py                 # call rate, heterozygosity, inferred sex
python scripts/02_traits.py [--apoe]    # single-SNP traits

# References: 1000 Genomes (~15 GB, verified + resumable) and ClinVar; safe to rerun
bash scripts/00_download_refs.sh        # or only 1000G: bash scripts/download_1kg.sh [chroms...]
python scripts/03_clinvar.py            # ClinVar carriers (confirm anything clinically!)
bash scripts/05_merge_1kg.sh            # sample + 1000G → outputs/merged.{pgen,pvar,psam} (runs 04_make_vcf.py)
bash scripts/06_roh.sh                  # runs of homozygosity
bash scripts/07_ancestry.sh             # PCA + supervised ADMIXTURE + plot (08_plot_ancestry.py)
python scripts/09_prs.py                # polygenic scores vs 1000G superpop
bash scripts/download_pharmcat.sh       # PharmCAT jar + GRCh38 positions (~50 MB) into ref_dir/pharmcat/
bash scripts/10_pharmcat.sh             # PharmCAT; Java: conda create -n java -c conda-forge --override-channels openjdk=21
python scripts/11_relatives.py --other relative_raw.csv --name mum

# CRAM
bash scripts/cram_qc.sh                 # idxstats, coverage, chr1 depth histogram
# then myheritage_comprehensive_exploratory_analysis.ipynb
```

## qpAdm with ancient and Eastern European references (AADR)

1000 Genomes has no Eastern Europeans apart from Finns, so these steps use the Allen Ancient DNA Resource
(Human Origins panel), which has modern Russian, Ukrainian and Volga-Uralic groups as well as ancient genomes.

```bash
bash scripts/download_aadr.sh                # AADR v66.p1 HO, ~4.1 GB, MD5-verified
python scripts/12_aadr_prepare.py check      # are the group labels in config `aadr.groups` real?
python scripts/12_aadr_prepare.py find Udmurt Yamnaya   # search labels to fix them
python scripts/12_aadr_prepare.py build      # extract groups + you (CRAM pass ~30+ min, reused afterwards)
python scripts/13_qpadm.py                   # all models in config `qpadm.analyses`, resumable
python scripts/20_qpadm_replicates.py pileup && python scripts/20_qpadm_replicates.py run -r 30   # repeat the random-read draw
python scripts/21_farmer_f4.py         # f4(Mbuti, Anatolia_N; EHG/WHG/Yamnaya, test) + admixture f3 for populations
python scripts/22_pca_ancient.py        # West Eurasian PCA, ancient genomes and you projected (lsqproject-style)
python scripts/23_qpadm_east.py --replicates 10   # which population carries the Siberian-related share (after 12_aadr_prepare.py build)
python scripts/24_neanderthal.py         # Neanderthal ancestry (f4-ratio) with control individuals
python scripts/25_shared_alleles.py       # alleles rare today but common in ancient groups: rates vs controls, annotated list
python scripts/26_g25_style.py           # G25-style coordinates (25 scaled PCs), distances, Vahaduo-style mixtures (~25 min)
python scripts/27_ibs_all.py             # Ancestrify-style IBS scan: you (GLIMPSE-imputed) + 25 controls vs all 14,743 ancients
python scripts/15_ibs_ancient.py             # GEDmatch-style ancient "matches" vs 1000G/AADR controls
python scripts/18_f3_ancient.py              # outgroup f3(Mbuti; you, X) for ~15k ancient individuals
python scripts/19_f4_uralic.py               # f4(Early Slavs, X; Ukrainian, you) vs Russian/Belarusian/Ukrainian controls
```

### Real IBD with ancient genomes (ancIBD)

GLIMPSE2 imputes and phases your CRAM and ~40 medieval Eastern European shotgun genomes against the local
1000 Genomes panel (genetic map from the AADR .snp), then ancIBD looks for shared segments ≥ 8 cM.

```bash
CLEAN_SPLIT=1 bash scripts/16_glimpse.sh outputs/glimpse/me.tsv me   # your CRAM, ~5-6 h, resumable
python scripts/17_ancibd.py select                 # 40 individuals from the AADR annotation (config `ancibd`)
bash scripts/download_ancient_bams.sh manifest     # ENA file lists -> outputs/ancibd/manifest.tsv
bash scripts/download_ancient_bams.sh fetch        # BAMs to D: (checks free space first)
bash scripts/download_ancibd.sh                    # ancIBD (own conda env) + its 1240k files
```
Keep large files on D:. WSL's own disk is a file on C:, which has little free space.

## Report

```bash
python scripts/14_report.py   # outputs/report/index.html, qpadm.html, figures/*.png
"/mnt/c/Program Files/Mozilla Firefox/firefox.exe" "$(wslpath -w outputs/report/index.html)"   # 14_report.py does this itself (--no-open to skip)
```

Self-contained HTML pages (light/dark, hover for details) built from whatever outputs exist so far;
rerun it after any step.

You enter qpAdm twice: `me_cram` (one random read per SNP from the CRAM, unaffected by imputation) and
`me_csv` (the imputed MyHeritage genotypes). Trust a model when it fits for both. `modern` asks whether
a Uralic/Volga source is needed on top of Slavic ones; `modern_wide` tries Baltic, Finnic, Volga-Turkic and
Central European groups (up to two at a time); `deep` gives WHG / EHG / Anatolia_N / Yamnaya / Nganasan
proportions; `proximal` uses closer-in-time Corded Ware and Early Slav sources. Run a subset with
`--analysis modern_wide proximal`; after adding groups to the config, rerun `12_aadr_prepare.py build`. Prefer the simplest model that fits (p ≥ 0.01, all weights in [0, 1]).

Shell scripts share `scripts/env.sh` (strict mode, config values as `$OUT`, `$REF`, `$THREADS`…);
Python scripts share `scripts/common.py`.

## Troubleshooting (WSL2)

- **`admixture` segfaults** right after `Size of G`: the 2018 static binary needs the legacy vsyscall page.
  Put `[wsl2]` / `kernelCommandLine = vsyscall=emulate` in `C:\Users\<you>\.wslconfig`, then `wsl --shutdown`.
- **PharmCAT input**: your genotypes are matched to PharmCAT's GRCh38 positions by rsID (no liftover, no
  Docker). The raw file has no indels and lacks some key SNPs (e.g. CYP2C19*2, CYP2C9*3, VKORC1), so those
  genes may come out as *1 by default; check "Positions with data" in the report. CYP2D6 is not callable.
- **Keep large downloads off `/mnt/d` until verified**: `download_1kg.sh` stages in `~/1kg_staging` and checks
  every file with `bgzip -t` before and after moving it.

Caveats: rare variants from imputed/array data are often wrong; PRS percentiles depend on ancestry match;
this is not a medical test.
