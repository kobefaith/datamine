import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 数据准备
N = 1000
x = np.random.randn(N)
y = np.random.randn(N)
# 用 Matplotlib 画散点图
plt.scatter(x, y,marker='o')
#plt.scatter(x, y,marker='x')
plt.show()
# 用 Seaborn 画散点图
df = pd.DataFrame({'x': x, 'y': y})
sns.jointplot(x="x", y="y", data=df, kind='scatter');
plt.show()
