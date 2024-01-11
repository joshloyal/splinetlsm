# Simulation and Real Data Analysis Scripts

Simulation and real data analysis scripts. The code was original written to run on a HPC cluster. 

## How to Run

The following commands run the simulations or real data analysis and produce the figures in the corresponding section.

### Section 6.2 (Parameter Recovery for Varying Network Sizes)

These commands run the simulations and produce Figure 1 and Figure 2.

```bash
>>> python simulation_recovery.py
>>> cd output_parameter_recovery/
>>> python process.py
>>> python plot_nodes.py
>>> python plot_time.py
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

### Appendix G (Sensitivity to Subsample Fractions)

The following commands runs the simulations and produce Figure 7(a).

```bash
>>> python simulation_nonedge_sensitivity.py
>>> cd output_nonedge_sensitivty/
>>> python process.py
>>> python plot.py
```

The following commands runs the simulations and produce Figure 7(b).

```bash
>>> python simulation_time_fraction_sensitivity.py
>>> cd output_time_fraction_sensitivty/
>>> python process.py
>>> python plot.py
```

### Section 7 and Appendix J (Weekly Conflict Networks)

To produce the figures, you will need to run the cells in the corresponding Jupyter notebook:

```bash
>>> jupyter notebook application_POLECAT.ipynb
```
