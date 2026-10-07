# DORTS-9K MLIP benchmark

Code and processed results for our benchmark of pretrained machine-learning interatomic potentials on the DORTS-9K reaction dataset. Fifteen checkpoints are compared with the ωB97M-V/def2-TZVP reference on all 909,077 configurations, and seventeen on the CHNO subset.

Interactive versions of the figures and Table 2 are at [austin243.github.io/dorts9k-mlip-benchmark](https://austin243.github.io/dorts9k-mlip-benchmark/). The Figure 5 page shows its reactions in 3D, together with six more CHNO reactions of each reaction class.

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
| `paper/figure4.py` | Figures 4 and S2, transition-state errors by reaction class |
| `paper/figureS3.py` | Figure S3, CHNO error tails |
| `paper/tables.py` | Table 2 and Tables S1 to S5 and S7 |
| `paper/interactive.py` | The interactive site in `results/site`, which GitHub Pages publishes |

The static Figure 5 is built from VESTA renderings of six reactions and is not generated here. The site shows the same reactions in 3D, together with six more CHNO reactions of each reaction class. Every quantity is defined in the [calculation notes](docs/calculations.md).

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
| `figure5_reactions.json` | Reactant, transition-state, and product geometries of the six Figure 5 reactions from DORTS-9K, aligned on the transition state, with their changing contacts and errors |
| `class_examples.json` | The same for six CHNO reactions of each reaction class, at the 10th, 50th, 75th, 90th, 95th, and 99th percentiles of the UMA-M OMol total barrier error within the class |
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
