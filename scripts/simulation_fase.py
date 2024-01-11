import os
import subprocess


out_dir = 'output_comparison_fase'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for density in [0.1, 0.2, 0.3]:
    for (n, M) in [(100, 10), (100, 20), (200, 10)]:
        for i in range(50):
            subprocess.run(f"python make_network.py {i} {n} {M} {density}".split(' '))
            subprocess.run(f"Rscript fase.R {i} {n} {M} {density}".split(' '))
