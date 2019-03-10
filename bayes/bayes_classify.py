#!/usr/bin/env python
# -*- coding:utf8 -*-
import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# word_list = jieba.cut (text) # 中文分词
# stop_words = [line.strip().decode('utf-8') for line in io.open('./text_classification/stop/stopwords.txt').readlines()]
# tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
# features = tf.fit_transform(train_contents)

# # 多项式贝叶斯分类器
# from sklearn.naive_bayes import MultinomialNB  
# clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
# test_tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5, vocabulary=train_vocabulary)
# test_features=test_tf.fit_transform(test_contents)
# predicted_labels=clf.predict(test_features)
# from sklearn import metrics
# print metrics.accuracy_score(test_labels, predicted_labels)


# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf_vec = TfidfVectorizer()
# documents = [
#     'this is the bayes document',
#     'this is the second second document',
#     'and the third one',
#     'is this the document'
# ]
# tfidf_matrix = tfidf_vec.fit_transform(documents)

# print('不重复的词:', tfidf_vec.get_feature_names())
# print('每个单词的 ID:', tfidf_vec.vocabulary_)
# print('每个单词的 tfidf 值:', tfidf_matrix.toarray())

def load_data(base_path):    
    documents = []
    labels = []
    for root, dirs, files in os.walk(base_path): # 循环所有文件并进行分词打标        
        for file in files:
            label = root.split('\\')[-1] # 因为windows上路径符号自动转成\了，所以要转义下
            labels.append(label)
            filename = os.path.join(root, file)
            with open(filename, 'rb') as f: # 因为字符集问题因此直接用二进制方式读取
                content = f.read()
                word_list = list(jieba.cut(content))                
                words = [wl for wl in word_list]
                documents.append(' '.join(words))
    return documents, labels

train_contents, train_labels = load_data('./text_classification/1/test')
test_contents, test_labels = load_data('./text_classification/1/test2')
stop_words = []
# with open('./text_classification/stop/stopword.txt') as f:
# 	for line in f.readlines():
# 		stop_words.append(line.strip().encode('utf-8'))
import io
stop_words = [line.strip().encode('utf-8') for line in io.open('./text_classification/stop/stopword.txt').readlines()]



tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
train_features = tf.fit_transform(train_contents)
# 多项式贝叶斯分类器
 
clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)

test_tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5, vocabulary=tf.vocabulary_)
test_features=test_tf.fit_transform(test_contents)
predicted_labels=clf.predict(test_features)
print (metrics.accuracy_score(test_labels, predicted_labels))









