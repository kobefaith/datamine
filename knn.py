# -*- coding: utf-8 -*
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
vector1 = np.array([1,2,3])
vector2 = np.array([4,5,6])
trainarr = np.array([[1,2,3],[4,5,6],[7,8,9]])

def cal_dist(vector1,vector2):
	return np.sqrt(np.sum(np.square(vector1-vector2)))
def knn_classify(target,train):
	classarr = 	np.array([])
	for index in range(len(train)):		
		np.append(classarr,[train[index][-1],cal_dist(target,train[index])])
	print (classarr)

knn_classify(vector1,trainarr)

