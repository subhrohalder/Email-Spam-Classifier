#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 13:44:59 2020

@author: subhrohalder
"""

#imports
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#importing the data
df_data = pd.read_csv('./dataset/emails.csv')

#getting text and spam columns
df_data = df_data[['text','spam']]

#checking spam column unique values count
df_data['spam'].value_counts()

#only taking data with spam = 0 or 1
df_data = df_data.loc[(df_data['spam']== '0') | (df_data['spam']== '1')]

df_data['spam'].value_counts()

#not spam dataframe
df_not_spam = df_data.loc[df_data['spam']== '0']

#spam data frame
df_spam = df_data.loc[df_data['spam']== '1']

#checking the percentage
print('Spam Percentage: ',(len(df_spam)/len(df_data)))

print('Not Spam Percentage: ',(len(df_not_spam)/len(df_data)))

#plotting the cont plot
sns.countplot(df_data['spam'], label = 'Count Spam Vs. Not Spam')

#example for count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

sample_data = ['This is first','This is second','This is third','This is fourth']

sample_vectorizer = CountVectorizer()

#transforming the data
X = sample_vectorizer.fit_transform(sample_data)

print(X.toarray())

print(sample_vectorizer.get_feature_names())

#implementation
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

spam_notspam_countvectorizer = vectorizer.fit_transform(df_data['text'])

print(vectorizer.get_feature_names())

print(spam_notspam_countvectorizer.toarray())

spam_notspam_countvectorizer_array = spam_notspam_countvectorizer.toarray()

label = df_data['spam'].values

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()

#fitting
NB_classifier.fit(spam_notspam_countvectorizer,label)

#testing
testing_data = ['You won 100$ course for free','Please find the below documents for verification']

testing_data_vectorizer = vectorizer.transform(testing_data)

NB_classifier.predict(testing_data_vectorizer)

#model building by splitting train test data
X = spam_notspam_countvectorizer
y = label

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


from sklearn.naive_bayes import MultinomialNB

NB_classifier_train_test = MultinomialNB()

NB_classifier_train_test.fit(X_train,y_train)

from sklearn.metrics import classification_report,confusion_matrix

#testing with train data
y_pred_train = NB_classifier_train_test.predict(X_train)

cm = confusion_matrix(y_train,y_pred_train)

sns.heatmap(cm,annot=True)

#testing with test data
y_pred_test = NB_classifier_train_test.predict(X_test)

cm = confusion_matrix(y_test,y_pred_test)

sns.heatmap(cm,annot=True)

print(classification_report(y_test,y_pred_test))
    
    