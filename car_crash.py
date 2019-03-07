import matplotlib.pyplot as plt
import seaborn as sns
# 数据准备
car_crash = sns.load_dataset('car_crashes')
# 用 Seaborn 画成对关系
sns.pairplot(car_crash)
plt.show()
