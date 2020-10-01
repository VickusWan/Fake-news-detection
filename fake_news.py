# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:29:05 2020

@author: Victor
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing

df = pd.read_csv(r"C:\Users\Victor\.spyder-py3\news.csv")
df_copy = df.copy()

le = preprocessing.LabelEncoder()
df_copy['label']=le.fit_transform(df_copy['label'])

n = df.isnull().sum()



for index, value in n.items():
    if value != 0:
        print(f"missing values inside:  {index}")
    else:
        df[index].fillna(df[index].mode()[0], inplace=True)
         
X = df.text
Y = df.label
 
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)

#================================
#Passive Aggressive Algorithm

model = PassiveAggressiveClassifier() 
model.fit(tfidf_train, y_train)
# Fitting model  
y_pred=model.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy PA: {round(score*100,2)}%')


#===============================

NB = BernoulliNB()
NB.fit(tfidf_train, y_train)
p = NB.predict(tfidf_test)
s = accuracy_score(y_test, p)
print(f'Accuracy Bernoulli: {round(s*100,2)}%')

#===============================

lr = LogisticRegression()
lr.fit(tfidf_train, y_train)
p =lr.predict(tfidf_test)
s = accuracy_score(y_test, p)
print(f'Accuracy Logistics: {round(s*100,2)}%')

#================================

# =============================================================================
# tree = DecisionTreeClassifier()
# tree.fit(tfidf_train, y_train)
# p =tree.predict(tfidf_test)
# s = accuracy_score(y_test, p)
# print(f'Accuracy Tree: {round(s*100,2)}%')
# =============================================================================

# =============================================================================
# clf = SVC()
# clf.fit(tfidf_train, y_train)
# p =clf.predict(tfidf_test)
# s = accuracy_score(y_test, p)
# print(f'Accuracy SVM: {round(s*100,2)}%')
# =============================================================================
