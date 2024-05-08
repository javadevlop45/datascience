import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Correct import
import seaborn as sb
import warnings
from scipy import stats

warnings.filterwarnings('ignore')
df = sb.load_dataset('mpg')
print(df)
print(df['horsepower'].describe())
print(df['model_year'].describe())
bins = [0, 75, 150, 240]
df['horsepower_new'] = pd.cut(df['horsepower'], bins=bins, labels=['l', 'm', 'h'])
c = df['horsepower_new']
print(c)
ybins = [69, 72, 74, 84]
labels = ['t1', 't2', 't3']  # Corrected variable name
df['modelyear_new'] = pd.cut(df['model_year'], bins=ybins, labels=labels)  # Corrected assignment
newyear = df['modelyear_new']  # Corrected variable name
print(newyear)
df_chi = pd.crosstab(df['horsepower_new'], df['modelyear_new'])
print(df_chi)
print(stats.chi2_contingency(df_chi))  # Corrected function name
