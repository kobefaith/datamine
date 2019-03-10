#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os 
import jieba 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn import metrics

def preprocess(path_name):
    text_with_spaces=""
    textfile=open(path_name,"rb").read() 
    textcut=jieba.cut(textfile)
    for word in textcut:
        text_with_spaces+=word+" "
    return text_with_spaces

def loadtrainset(path,classtag):
    allfiles=os.listdir(path)
    processed_textset=[]
    allclasstags=[]
    for thisfile in allfiles:
        path_name=path+"/"+thisfile
        processed_textset.append(preprocess(path_name))
        allclasstags.append(classtag)
    return processed_textset,allclasstags

processed_textdata1,class1=loadtrainset("./text_classification/train/女性", "女性")
processed_textdata2,class2=loadtrainset("./text_classification/train/体育", "体育")
processed_textdata3,class3=loadtrainset("./text_classification/train/文学", "文学")
processed_textdata4,class4=loadtrainset("./text_classification/train/校园", "校园")
integrated_train_data=processed_textdata1+processed_textdata2+processed_textdata3+processed_textdata4
classtags_list=class1+class2+class3+class4
count_vector = CountVectorizer()

vector_matrix = count_vector.fit_transform(integrated_train_data)
train_tfidf = TfidfTransformer(use_idf=False).fit_transform(vector_matrix)
clf = MultinomialNB().fit(train_tfidf,classtags_list)#

test_textdata1,testClass1=loadtrainset("./text_classification/test/女性", "女性")
test_textdata2,testClass2=loadtrainset("./text_classification/test/体育", "体育")
test_textdata3,testClass3=loadtrainset("./text_classification/test/文学", "文学")
test_textdata4,testClass4=loadtrainset("./text_classification/test/校园", "校园")
integrated_test_data=test_textdata1+test_textdata2+test_textdata3+test_textdata4
classtags_list=testClass1+testClass2+testClass3+testClass4
new_count_vector = count_vector.transform(integrated_test_data)
new_tfidf= TfidfTransformer(use_idf=False).fit_transform(new_count_vector)
predict_result = clf.predict(new_tfidf) 

print(metrics.accuracy_score(classtags_list, predict_result))