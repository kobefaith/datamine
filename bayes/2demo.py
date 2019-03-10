#!/usr/bin/env python
# -*- coding:utf8 -*-

import os
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# 1. 加载数据
# 加载停用词表
l_stopWords = set()
with open('./text_classification/stop/stopword.txt', 'r') as l_f:
    for l_line in l_f:
        l_stopWords.add(l_line.strip())

l_labelMap = {'体育': 0, '女性': 1, '文学': 2, '校园': 3}
# 加载训练数据和测试数据
def LoadData(filepath):
    l_documents = []
    l_labels = []
    for root, dirs, files in os.walk(filepath):
        for l_file in files:
            l_label = root.split('/')[-1]
            l_filename = os.path.join(root, l_file)
            
            with open(l_filename, 'rb') as l_f:
                l_content = l_f.read()
                l_wordlist = list(jieba.cut(l_content))
                l_words = [item for item in l_wordlist if item not in l_stopWords]
                l_documents.append(' '.join(l_words))
                l_labels.append(l_labelMap[l_label])
                
    return l_documents, l_labels

l_trainDocuments, l_trainLabels = LoadData('./text_classification/train')
l_testDocuments, l_testLabels = LoadData('./text_classification/test')

# # 2. 计算权重矩阵
l_tfidfVec = TfidfVectorizer(max_df=0.5)
l_tfidfMatrix = l_tfidfVec.fit_transform(l_trainDocuments)

# for item in l_tfidfVec.get_feature_names():
# print item
# print l_tfidfVec.get_feature_names()
# print l_tfidfVec.vocabulary_
# print (l_tfidfMatrix.toarray().shape)

# # 3. 朴素贝叶斯模型
# ## 3.1 模型训练
l_clf = MultinomialNB(alpha=0.001)
l_clf.fit(l_tfidfMatrix, l_trainLabels)

# ## 3.2 模型预测
l_testTfidf = TfidfVectorizer(max_df=0.5, vocabulary=l_tfidfVec.vocabulary_)
l_testFeature = l_testTfidf.fit_transform(l_testDocuments)
l_hats = l_clf.predict(l_testFeature)

# ## 3.3 模型评估
from sklearn.metrics import accuracy_score
print (accuracy_score(l_hats, l_testLabels))