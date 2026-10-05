# Timing settings

Figure 2 shows the energy and force evaluation time of the fifteen full-dataset checkpoints on NVIDIA A100-SXM4-80GB GPUs. The Python checkpoints ran with PyTorch 2.8.0 and CUDA 12.8. The timer includes tensor and graph preparation inside each model adapter and the transfer of results back to the CPU. It excludes model construction, HDF5 decoding, error analysis, and output writing. NEP89 runs as a GPUMD `nep` subprocess, so its time also includes process start-up, model loading, and prediction file I/O. First calls are kept, with no warm-up subtraction.

Each configuration is assigned its batch's measured time divided by the number of configurations in the batch. The mean time per configuration is 1000 × (sum of configuration times) / (number of configurations) in ms. Times from parallel workers are summed and not divided by the number of GPUs, so the value measures production throughput rather than single-configuration latency or speed at a common batch size. Retries of failed batches are not counted. The NEP89 time comes from the rerun with the corrected prediction input described in the [environment notes](../environments/README.md#nep89).

ANI-1xnr was timed separately on all 399,750 CHNO configurations because its accuracy run also computed Hessians. That measurement used 1,580 batches on eight workers and took 42.782144258 summed seconds, or 0.10702224955 ms per configuration. It is not part of Figure 2.

## Production settings

| Family | Precision and settings | Normal batch → large-molecule batch |
| --- | --- | --- |
| MACE-MH-1 | float64 with the OMol, OMat-PBE, or SPICE-wB97M head | 256 → 128 at N≥24 |
| MACE-POLAR-1 S / M / L | float64 | 64 / 32 / 16 → 32 / 16 / 8 at N≥24 |
| UMA S | float32, `fast_precise` | 32 → 16 at N≥22 |
| UMA M | float32, `fast_precise` | 16 → 8 at N≥18 |
| OrbMol-v2 | float32-high with fp64 energy | 128 → 64 at N≥24 |
| SevenNet-i12 | float32 with FlashTP | 64 → 32 at N≥32 |
| AIMNet2 and AIMNet2-RXN | float32 inputs, ensemble member 0 | 256 → 128 at N≥32 |
| ANI-1xnr timing | float64, eight-member ensemble, pyaev | 256 |
| NEP89 | GPUMD v5.0 `nep` | 4,096, at most 160,000 atoms |

The Python runners use eight CPU threads, and the ANI-1xnr timing run uses one. UMA's `fast_precise` setting is the turbo profile with TF32, graph merging, compilation, and Hessian prediction disabled. Charge 0 and spin multiplicity 1 are passed to MACE-POLAR-1, OrbMol-v2, the UMA OMol head, AIMNet2, and AIMNet2-RXN. The UMA OMC head receives charge 0 with the closed-shell spin feature 0. The other checkpoints take no charge or spin input. The batch settings are also recorded in `src/dorts9k/models.py`.

## Repeating the ANI-1xnr timing

`scripts/time_ani.py` reproduces the whole-batch partitioning. Each worker takes the global batch numbers equal to its index modulo eight and keeps the configuration order within each batch. Run it in a fresh directory with one visible GPU per worker (see the [Slurm example](../scripts/time_ani.slurm)), then combine the workers.

```bash
python scripts/combine_ani_timing.py results/ani_timing
```

The combination step rejects missing or duplicated batches, sums the single-GPU times, and writes `combined.json` next to the worker files. The general `dorts9k evaluate` command also records ANI-1xnr times, but it batches configurations by shard, which differs from this protocol.
