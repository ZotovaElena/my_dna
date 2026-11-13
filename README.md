# Personal DNA analysis (MyHeritage raw file)

Input: `data/MyHeritage_raw_dna_data.csv` — ~585k SNPs, GRCh37 forward strand, low-pass WGS + imputation,
autosomes + X only (no Y / MT, so no haplogroups). Settings in `configs/config.yaml`; outputs in `outputs/`,
downloaded references in the configured `ref_dir`.

```bash
mamba env create -f environment.yml && conda activate dna

# No references needed
python scripts/01_qc.py                 # call rate, heterozygosity, inferred sex
python scripts/02_traits.py [--apoe]    # single-SNP traits
python scripts/12_gwas_catalog.py       # GWAS Catalog associations for configured traits
python scripts/12_gwas_catalog.py --rsid rs12913832 rs4988235  # selected rsIDs
python scripts/12_gwas_catalog.py --refresh  # re-download instead of using outputs/gwas_cache/
python scripts/13_plot_gwas_catalog.py [--top 20 --per-rsid 3]  # outputs/gwas_catalog_associations.png
python scripts/14_gwas_trait_map.py [--per-category 5]  # trait categories + traits linked to alleles you carry
python scripts/15_gwas_explorer.py      # outputs/gwas_explorer.html: offline, filterable table of all associations

# References (~15 GB; stored at C:\Users\ezotova\Documents\MY\myheritage in Windows)
bash scripts/00_download_refs.sh
python scripts/03_clinvar.py            # ClinVar carriers (confirm anything clinically!)
bash scripts/05_merge_1kg.sh            # sample + 1000G → outputs/merged.{pgen,pvar,psam}
bash scripts/06_roh.sh                  # runs of homozygosity
bash scripts/07_ancestry.sh             # PCA + supervised ADMIXTURE + plot
python scripts/09_prs.py                # polygenic scores vs 1000G superpop
bash scripts/10_pharmcat.sh             # GRCh38 liftover + PharmCAT (docker)
python scripts/11_relatives.py --other relative_raw.csv --name mum
```

Caveats: rare variants from imputed/array data are often wrong; PRS percentiles depend on ancestry match;
GWAS associations are population-level findings, not personal medical predictions. The GWAS Catalog
script sends rsIDs only (not genotypes) to its public API. These analyses are not medical tests.
