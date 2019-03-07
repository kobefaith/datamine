# -*- coding: utf-8 -*
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
data = {'chinese': [66, 95, 95,90, 80,80],'english': [65, 85, 92, 88, 90,90],'math': [ np.nan,98, 96, 77, 90,90]}
df2 = DataFrame(data, index=['zhangfei', 'guanyu', 'zhaoyun', 'huangzhong', 'dianwei','dianwei'], columns=['chinese', 'english', 'math'])
df2.drop_duplicates(inplace=True)
print (df2)
df2.rename(columns={'chinese': '语文', 'english': '英语','math':'数学'},index={'zhangfei': '张飞', 'guanyu': '关羽','zhaoyun':'赵云','huangzhong':'黄忠','dianwei':'典韦'}, inplace = True)
df2.isnull()
df2.loc['张飞','数学']=0
df2['总和'] = df2['语文']+df2['英语']+df2['数学']

print (df2)


