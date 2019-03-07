# 数据准备
# 生成 0-1 之间的 10*4 维度数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=np.random.normal(size=(10,4)) 
lables = ['A','B','C','D']
# 用 Matplotlib 画箱线图
plt.boxplot(data,labels=lables)
plt.show()
# 用 Seaborn 画箱线图
df = pd.DataFrame(data, columns=lables)
sns.boxplot(data=df)
plt.show()
