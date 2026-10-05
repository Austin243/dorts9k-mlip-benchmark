# Calculation notes

These definitions follow Section 3 of the paper. `dorts9k summarize` applies them to the saved predictions, and the scripts in `paper/` aggregate its output.

**Reactions and configurations.** DORTS-9K has 9,017 reactions and 909,077 configurations. The CHNO subset has 3,995 reactions and 399,750 configurations whose atoms are all C, H, N, or O. A direct anchor is an explicitly stored reactant, product, transition-state, or maximum-slope geometry. Not every reaction has all five direct anchors, so each comparison uses the reactions that have the geometries it needs. Every checkpoint is evaluated on the same set.

| Comparison | Full dataset | CHNO subset |
| --- | --- | --- |
| Global force RMSE | 909,077 configurations, 9,017 reactions | 399,750 configurations, 3,995 reactions |
| Global relative energy | 865,669 configurations, 8,493 reactions | 376,776 configurations, 3,713 reactions |
| Transition-state force RMSE | 8,768 reactions | 3,871 reactions |
| Total barrier error | 8,493 reactions | 3,713 reactions |
| Five-anchor profiles | 8,474 reactions | 3,706 reactions |

**Force.** The force RMSE of a configuration is the root mean square over its 3N Cartesian components, in eV/Å. Global statistics pool all configurations. Transition-state (TS) statistics use the one direct TS anchor of each reaction. A mean or median of configuration RMSEs is not a pooled RMSE over all components.

**Reference endpoint.** For configuration i, the residual is d_i = E_model,i − E_DFT,i in eV. The reference endpoint of a reaction is the explicit reactant or product with the lower DFT energy, and the reactant on an exact tie. The same endpoint geometry is used for the checkpoint and for DFT. A reaction without both explicit endpoints has no relative-energy or barrier errors.

**Relative energy.** The global relative-energy error of configuration i is 1000 |d_i − d_ref| / N_i in meV/atom. N_i counts all atoms, including hydrogen. Global medians are taken over configurations.

**Total barrier error.** At the TS anchor the total barrier error is c (d_TS − d_ref) in kcal/mol, with c = 23.06054783061903 kcal/mol per eV and no division by the atom count. A negative value means the checkpoint underestimates the reference energy difference. The figures and most tables use its absolute value, and total barrier MAE averages that over reactions. The reference barrier is c (E_DFT,TS − E_DFT,ref). Table 2 divides each MAE by the mean reference barrier of the 3,713 CHNO reactions, 62.9 kcal/mol.

**Profiles.** Figures 3 and S1 use the reactions with all five direct anchors. Forces are shown at all five. The reference endpoint has zero relative-energy error by construction, so the energy columns are the absolute total errors c |d_j − d_ref| at the reverse maximum-slope anchor, the TS, and the forward maximum-slope anchor, followed by the reaction-energy error c |d_P − d_R|. That last quantity equals the absolute error in E_P − E_R. Each error is converted to kcal/mol before the median across reactions is taken.

**Distributions and tails.** The ECDFs in Figures 4 and S2 keep every error, and an axis limit only crops the drawing. Quantiles use linear interpolation. The worst 1% of a checkpoint holds its largest ceil(0.01 n) errors, which is 39 TS force errors or 38 barrier errors on the CHNO subset. Its error share is their sum as a percentage of the sum over all reactions.

**Confidence intervals.** Table S3 gives pointwise percentile 95% intervals from 2,000 resamples of reactions with NumPy's `default_rng(20260909)`. Within a metric, all checkpoints share each resampled set of reactions. The force metric is resampled first and the barrier metric second from the same random stream. The intervals describe the variability from resampling reactions and not the uncertainty of the DFT reference.
