# Simulation and Real Data Analysis Scripts

Simulation and real data analysis scripts. The code was original written to run on a HPC cluster. 

## How to Run

The following commands run the simulations or real data analysis and produce the figures in the corresponding section.

### Section 6.2 and Appendix G.2 (Parameter Recovery for Varying Network Sizes)

These commands run the simulations and produce Figure 1, Figure 2, Figures S.1 through S.4, and calculate the percentage that the latent space dimension is correctly estimated.

```bash
>>> python simulation_recovery.py
>>> cd output_parameter_recovery/
>>> python process.py
>>> python plot_nodes.py
>>> python plot_time.py
>>> python plot_selection.py
>>> python plot_selection_time.py
```

### Section 6.3 (Method Comparison)

These commands run the simulations and produces Table 1.

```bash
>>> python simulation_comparison.py
>>> python simulation_fase.py
>>> cd output_comparison/
>>> python process.py
>>> python print_results.py
```

### Appendix G.3 (Comparison of SVI and MCMC Based Credible Intervals)

These commands run the simulations and produces Figures S.5 and S.6.

```bash
>>> python simulation_mcmc_vi_comparison.py
>>> cd output_mcmc_vi_comparison/
>>> python process.py
>>> python plot.py
```

### Appendix G.4 (Sensitivity to Subsample Fractions)

The following commands run the simulations and produce Figure S.7(a).

```bash
>>> python simulation_nonedge_sensitivity.py
>>> cd output_nonedge_sensitivty/
>>> python process.py
>>> python plot.py
```

The following commands run the simulations and produce Figure S.7(b).

```bash
>>> python simulation_time_fraction_sensitivity.py
>>> cd output_time_fraction_sensitivty/
>>> python process.py
>>> python plot.py
```

### Appendix G.5 (Parameter Recovery for Sparse Networks)

The following commands run the simulations and produce Figure S.8.

```bash
>>> python simulation_sparsity.py
>>> cd output_sparsity/
>>> python process.py
>>> python plot.py
```

### Appendix G.6 (Parameter Recovery Under Different Smoothness Settings)

The following commands run the simulations and produce Figure S.9.

```bash
>>> python simulation_smoothness.py
>>> cd output_smoothness/
>>> python process.py
>>> python plot.py
```
To produce the visualizations in Figures S.10 through S.14, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook Simulation\ Visualization.ipynb
```

### Appendix G.7 (Performance of the WAIC in Selecting the Basis Dimension $\ell$)

The following commands run the simulations and produce Figure S.15.

```bash
>>> python simulation_knot_selection.py
>>> cd output_knot_selection/
>>> python process.py
>>> python plots.py
```

### Appendix G.8 (Performance When d < d_0)

The following commands run the simulations, produce Figure S.16, and calculate the percentage that the latent space dimension is saturated.

```bash
>>> python simulation_underspecified_dimension.py
>>> cd output_underspecified_dimension/
>>> python process.py
>>> python plots.py
>>> python plot_selection.py
```

### Appendix G.9 (Sensitivity to the Fractional Power $\alpha$)

The following commands run the simulations and produce Figures S.17 through S.19.

```bash
>>> python simulation_alpha.py
>>> cd output_alpha/
>>> python process.py
>>> python plots.py
```

### Section 7 and Appendix G.10 and Appendix G.11 (Weekly Conflict Networks)

To produce the figures, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook application_POLECAT/POLECAT.ipynb
```

In addition, a video of the animated latent space is located at `application_POLECAT/ls_video.mp4`.

### Appendix G.12 (The Effect of Irregularly Spaced Time Points on the Real Data Application)

To produce the figures, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook application_POLECAT/Subsample\ Time\ Points.ipynb
```
