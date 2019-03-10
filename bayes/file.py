#!/usr/bin/env python
# -*- coding:utf8 -*-
import os 
import jieba 

def preprocess(path_name):
    text_with_spaces=""
    textfile=open(path_name,"r").read() 
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

processed_textdata1,class1=loadtrainset("./text_classification/1/test", "女性")
processed_textdata2,class1=loadtrainset("./text_classification/1/test2", "女性")
integrated_train_data=processed_textdata1+processed_textdata2
print (integrated_train_data)


def load_data(base_path):
    """
    :param base_path: 基础路径
    :return: 分词列表，标签列表
    """
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

train_documents, train_labels = load_data('./text_classification/1')
# print(train_documents)
import io
stop_words = [line.strip().encode('utf-8') for line in io.open('./text_classification/stop/stopword.txt').readlines()]
# print(stop_words)

