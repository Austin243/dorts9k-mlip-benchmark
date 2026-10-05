# Environments and checkpoints

Use **separate Python 3.12 environments for each model family**. The text files pin the principal packages observed in the production runs. They are not exhaustive platform lockfiles. GPU specifications target Linux/CUDA 12.8 and have not been reinstalled or GPU-tested during repository packaging.

For example, create and activate a Python 3.12 environment, then install one family and the package.

```bash
pip install -r environments/mace.txt
pip install .
```

Replace `mace.txt` with `uma.txt`, `orbmol.txt`, `sevennet.txt`, `aimnet.txt`, or `ani.txt`. The analysis, figures, and tables need only `analysis.txt` and no CUDA installation.

MACE-POLAR also requires `graph_electrostatics`. The environment includes [the upstream-recommended v0.4.0](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html). Its version was not recorded in the saved benchmark metadata.

| Runtime | Evaluated release / checkpoint | Upstream |
| --- | --- | --- |
| MACE 0.3.16 | MACE-MH-1, MACE-POLAR-1 S/M/L | [MACE foundations](https://github.com/ACEsuit/mace-foundations/releases) |
| FAIRChem 2.17.0 | `uma-s-1p2`, `uma-m-1p1`, OMol/OMC heads | [UMA](https://huggingface.co/facebook/UMA) |
| Orb models 0.7.0 | `orbmol-v2-teqabfhg-20260523.ckpt` | [Orb models](https://github.com/orbital-materials/orb-models) |
| SevenNet 0.13.0 | `checkpoint_sevennet_omni_i12.pth`, `omol25_low` / `spice` | [SevenNet](https://github.com/MDIL-SNU/SevenNet) |
| AIMNet 0.2.0 | `aimnet2_wb97m_d3_0.pt`, `aimnet2_rxn_0.pt` | [AIMNet](https://github.com/isayevlab/aimnetcentral) |
| TorchANI 2.8.2 | ANI-1xnr NeuroChem eight-member bundle | [ANI-1xnr](https://github.com/atomistic-ml/ani-1xnr) |
| GPUMD v5.0 | `nep89_20250409.txt` | [GPUMD](https://github.com/brucefan1983/GPUMD), [model tutorials](https://github.com/brucefan1983/GPUMD-Tutorials) |

Pass checkpoint paths explicitly. MACE's cached POLAR filenames may omit punctuation (`MACEPOLAR1Smodel`, etc.). UMA loads its named pretrained release through FAIRChem and checks that the supplied checkpoint matches it. Upstream access may require authentication. ANI needs the `.info` file **and** its referenced parameter, self-energy, and eight network directories in their original layout. Do not substitute a single ensemble member or a newer model release.

## SevenNet accelerator

The production run used FlashTP-e3nn 0.1.0. After installing `sevennet.txt`, build the [v0.1.0 release](https://github.com/SNU-ARC/flashTP/tree/v0.1.0) with a CUDA toolkit, following the [SevenNet instructions](https://github.com/MDIL-SNU/SevenNet/blob/main/docs/source/user_guide/accelerator.md).

```bash
git clone --branch v0.1.0 --depth 1 https://github.com/SNU-ARC/flashTP.git
cd flashTP
pip install -r requirements.txt
CUDA_ARCH_LIST=80 pip install . --no-build-isolation
```

Return to the benchmark repository to run evaluation. `80` targets A100. The evaluator requests FlashTP and does not silently replace it with a different accelerator.

## NEP89

Install `nep.txt` and the benchmark package. Build the `nep` prediction executable from [GPUMD v5.0](https://github.com/brucefan1983/GPUMD/releases/tag/v5.0) using the upstream CUDA build instructions, then pass `--nep-binary /path/to/GPUMD/src/nep`. This benchmark uses the official executable, not a Python NEP calculator.

GPUMD's prediction mode reads the network architecture from `nep.in`, not from the checkpoint, and falls back to its defaults for anything `nep.in` omits. The evaluator therefore copies the cutoffs, `n_max`, `basis_size`, `l_max`, neuron count, and ZBL radius from the `nep4_zbl` checkpoint header into `nep.in`. It stores that exact input with every output and refuses to resume from outputs written without it. The NEP89 results in `data/` come from a full rerun with this corrected input.
