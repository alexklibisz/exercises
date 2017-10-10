import pandas as pd
import matplotlib.pyplot as plt
from sys import argv

df = pd.read_csv(argv[1])
for c in df.columns:
    if c != 'epoch':
        plt.plot(df[c].values, label=c)
plt.legend()
plt.show()
