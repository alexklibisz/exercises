import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sys import argv

assert len(argv) == 3

df = pd.read_csv(argv[1])
plt.figure(figsize=(10, 12))
for c in sorted(df.columns):
    if c != 'epoch':
        plt.plot(df[c].values, label=c)
plt.xticks(np.arange(len(df)), np.arange(len(df)))
plt.xlabel('Epoch')
plt.title('Adversarial segmentation metrics')
plt.legend()
plt.savefig(argv[2], dpi=100)
