#!/usr/bin/env python
# coding: utf-8

# In[57]:


import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[58]:


df=pd.read_csv('NEWS.csv')

# Get the shape
df.shape


# In[59]:


df.head()


# In[60]:


df.loc[(df['label'] == 1) , ['label']] = 'FAKE'
df.loc[(df['label'] == 0) , ['label']] = 'REAL'


# In[61]:


labels = df.label
labels.head()


# In[62]:


x_train,x_test,y_train,y_test=train_test_split(df['text'].values.astype('str'), labels, test_size=0.2, random_state=7)


# In[63]:


#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


# In[64]:


# Fit & transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[69]:


#Passiveaggresiveclassifier
score11=[0]*4
for max_iter in range (1,4):
    pac1=PassiveAggressiveClassifier(max_iter)


# In[71]:


pac1.fit(x_train,y_train)
y_pred=pac1.predict(x_test)
score11[max_iter]=accuracy_score(y_test,y_pred)
print(max_iter,end=" 1")
print(f'Accuracy: {round(score11[max_iter]*2,2)}%')


# In[ ]:


print(" ACCURACY")
print("The highest value of accuracy is at max-iter of value " ,end="2")
print(score11.index(max(score11))) #print the value of iter where value is max
print('ACCURACY IS ',end="4")
print(round(max(score11)*100),end=" %")


# In[ ]:


print(y_test[0:10])
print(y_pred[0:10])


# In[67]:


print(x_test[0:5])


# In[ ]:




