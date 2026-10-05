# DORTS-9K MLIP benchmark

Code and processed results for our benchmark of pretrained machine-learning interatomic potentials on the DORTS-9K reaction dataset. Fifteen checkpoints are compared with the ωB97M-V/def2-TZVP reference on all 909,077 configurations, and seventeen on the CHNO subset.

Interactive versions of the data figures are at [austin243.github.io/dorts9k-mlip-benchmark](https://austin243.github.io/dorts9k-mlip-benchmark/).

## Figures and tables

The scripts in `paper/` rebuild the data figures and tables of the paper from the processed results in `data/`. Run them from the repository root after installing the analysis environment. They write to `results/figures` and `results/tables` and use the Arial font.

```bash
pip install -r environments/analysis.txt
python paper/figure1.py
```

| Script | Output |
| --- | --- |
| `paper/figure1.py` | Figure 1, global error distributions |
| `paper/figure2.py` | Figure 2, accuracy against evaluation time |
| `paper/figure3.py` | Figures 3 and S1, errors at the five direct anchors |
| `paper/figure4.py` | Figure 4, transition-state errors by reaction class |
| `paper/figureS2.py` | Figure S2, the CHNO version of Figure 4 |
| `paper/figureS3.py` | Figure S3, CHNO error tails |
| `paper/tables.py` | Table 2 and Tables S1 to S5 and S7 |
| `paper/interactive.py` | The interactive figure pages in `results/site`, which GitHub Pages publishes |

Figure 5 is built from VESTA renderings of six reactions and is not generated here. Every quantity is defined in the [calculation notes](docs/calculations.md).

## Data

| File | Contents |
| --- | --- |
| `ts_errors.csv.gz` | One row per checkpoint and reaction with the TS force RMSE, the absolute total barrier error, the reference endpoint, and the DFT reference barrier |
| `global_statistics.csv` | Force RMSE and relative-energy error statistics pooled over configurations, for the full dataset and the CHNO subset |
| `profile_statistics.csv` | Mean and quartiles of the errors at the five direct anchors, over the reactions that have all five |
| `timing.csv` | Mean evaluation time per configuration of the fifteen full-dataset checkpoints |
| `largest_uma_barrier_errors_signed.csv` | Signed total barrier errors of all seventeen checkpoints on the 38 reactions with the largest UMA-M OMol errors (Table S5) |
| `checkpoints.csv` | Checkpoint files, SHA-256 hashes, heads, precision, and package versions (Table S9) |
| `nep89_checks.csv` | Independent checks of the NEP89 inference (Table S10) |
| `reaction_manifest.csv` | Reaction identifiers, formulas, templates, and reaction classes |

The per-configuration model predictions are not included. DORTS-9K itself is available from [Zenodo](https://doi.org/10.5281/zenodo.17141108).

## Evaluation

The `dorts9k` package holds the code that produced the predictions. `dorts9k prepare` indexes the DORTS-9K HDF5 file and `dorts9k evaluate` runs one checkpoint in its own environment, as described in the [environment notes](environments/README.md) and the [Slurm example](scripts/evaluate.slurm). `dorts9k summarize` then reduces the saved shards to the files in `data/`. The timing protocol is in the [timing notes](docs/timing.md).

```bash
pip install .
dorts9k models
dorts9k prepare --source DORTS-9K.h5
dorts9k evaluate --model mace_mh1_omol --source DORTS-9K.h5 --checkpoint mace-mh-1.model
dorts9k summarize --runs runs --manifest prepared/manifest.h5
```
