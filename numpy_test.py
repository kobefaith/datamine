# -*- coding: utf-8 -*

import numpy as np
persontype = np.dtype({
    'names':['name', 'chinese', 'english', 'math','total'],
    'formats':['S32','i', 'i', 'i','i']})
peoples = np.array([("zhangfei",66,65,30,0),("guanyu",95,85,98,0),
       ("zhaoyun",93,92,96,0),("huangzhong",90,88,77,0),("dianwei",80,90,90,0)],
    dtype=persontype)
chineses = peoples[:]['chinese']
maths = peoples[:]['math']
englishs = peoples[:]['english']
print('========各科平均成绩==========')
print('语文的平均成绩是 ',np.mean(chineses))
print('数学的平均成绩是 ',np.mean(maths))
print('英语的平均成绩是 ',np.mean(englishs))
print('=========各科最小成绩=========')
print('语文的最小成绩是 ',np.amin(chineses))
print('数学的最小成绩是 ',np.amin(maths))
print('英语的最小成绩是 ',np.amin(englishs))
print('=========各科最大成绩=========')
print('语文的最大成绩是 ',np.amax(chineses))
print('数学的最大成绩是 ',np.amax(maths))
print('英语的最大成绩是 ',np.amax(englishs))
print('=========各科成绩标准差=========')
print('语文成绩标准差是 ',np.std(chineses))
print('数学成绩标准差是 ',np.std(maths))
print('英语成绩标准差是 ',np.std(englishs))
print('=========各科成绩方差=========')
print('语文成绩方差是 ',np.var(chineses))
print('数学成绩方差是 ',np.var(maths))
print('英语成绩方差是 ',np.var(englishs))
peoples[:]['total']= np.add(chineses,maths)
peoples[:]['total'] =np.add(peoples[:]['total'],englishs)
print (np.sort(peoples,order='total')[::-1])









