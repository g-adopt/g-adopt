#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

base = Path(__file__).parent.resolve()

df = pd.read_csv(base / "params.log", sep=r"\s+", header=0)
# pandas >= 3 reads the string column labels as its Arrow-backed ``str`` dtype
# when pyarrow is installed, which bakes a pyarrow dependency into the pickle
# that the CI image (no pyarrow) cannot unpickle. Store plain numpy ``object``
# labels so the reference loads anywhere and matches what CI's read_csv produces.
df.columns = df.columns.astype(object)
df.to_pickle(base / "expected.pkl")
print(df.to_string())
