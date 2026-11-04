# Personal DNA analysis (MyHeritage raw file)

Input: `MyHeritage_raw_dna_data.csv` — ~585k SNPs, GRCh37 forward strand, low-pass WGS + imputation,
autosomes + X only (no Y / MT, so no haplogroups). Settings in `configs/config.yaml`; outputs in `outputs/`,
downloaded references in the configured `ref_dir`.

```bash
mamba env create -f environment.yml && conda activate dna

# No references needed
python scripts/01_qc.py                 # call rate, heterozygosity, inferred sex
python scripts/02_traits.py [--apoe]    # single-SNP traits

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
this is not a medical test.
