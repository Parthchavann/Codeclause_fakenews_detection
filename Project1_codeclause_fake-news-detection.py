#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Loading necessary libraries and reading the file

# In[ ]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


#Read the data
df=pd.read_csv('news.csv')
#Get shape and head
df.shape
df.head()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


# Get the labels
labels=df.label
labels.head()


# ### Splitting dataset into training and test set

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[ ]:


#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)


# #### Let’s initialize a TfidfVectorizer with stop words from the English language and a maximum document frequency of 0.7 (terms with a higher document frequency will be discarded). Stop words are the most common words in a language that are to be filtered out before processing the natural language data. And a TfidfVectorizer turns a collection of raw documents into a matrix of TF-IDF features.
# 
# #### Now, fit and transform the vectorizer on the train set, and transform the vectorizer on the test set.

# In[ ]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# ### Next, initializing a PassiveAggressiveClassifier. This is. We’ll fit this on tfidf_train and y_train.
# 
# ### Then, we’ll predict on the test set from the TfidfVectorizer and calculate the accuracy with accuracy_score() from sklearn.metrics.

# In[ ]:


pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# ### Got  accuracy of 92.82% with this model. Finally, let’s print out a confusion matrix to gain insight into the number of false and true negatives and positives.

# In[ ]:


confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# ### So with this model, we have 591 true positives, 585 true negatives, 44 false positives, and 47 false negatives.

#  ### If you want to learn more about TfidfVectorizer , then you can refer [here .](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

# In[ ]:




