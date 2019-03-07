
# -*- coding: utf-8 -*
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
df = pd.read_csv('./fooddata.csv')
# 把ounces 列中的NaN替换为平均值
df['ounces'].fillna(df['ounces'].mean(), inplace=True)
# 把food列中的大写字母全部转换为小写
df['food'] = df['food'].str.lower()
# 把ounces 列中的负数转化为正数
df['ounces']= df['ounces'].apply(lambda x: abs(x))
df.drop_duplicates('food',inplace=True)
print (df)
