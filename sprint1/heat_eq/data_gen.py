import numpy as np
import pandas as pd

N = 500

x = np.random.rand(N)
t = np.random.rand(N)

u = np.exp(-np.pi**2 * t) * np.sin(np.pi * x)

data = pd.DataFrame({
    "x": x,
    "t": t,
    "u": u
})

data.to_csv("heat_data.csv", index=False)
print("Saved heat_data.csv")