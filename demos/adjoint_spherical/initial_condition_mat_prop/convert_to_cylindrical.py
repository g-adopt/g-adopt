import numpy as np
from pathlib import Path

for file in Path.cwd().rglob('*.txt'):
    r, profile = np.loadtxt(file, delimiter=',', unpack=True)
    if r[0] != 2.208:
        raise Exception()
    new_r = r / r[0] * 2.2
    output_str = "\n".join([f'{radius:.4f}, {value:.4f}' for radius, value in zip(new_r, profile)])
    with open(Path(str(file).replace(".txt", "_cyl.txt")), mode="w") as f:
        f.write(output_str)
