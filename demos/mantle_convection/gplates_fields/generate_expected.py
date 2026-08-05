#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

base = Path(__file__).parent.resolve()

df = pd.read_csv(base / "params.log", sep=r"\s+", header=0)
df.to_pickle(base / "expected.pkl")
print(df.to_string())
