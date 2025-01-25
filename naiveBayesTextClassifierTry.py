#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading necessary libraries and environment variables
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups

# Fetching the dataset
data = fetch_20newsgroups()

# Displaying the target names
data.target_names


# In[2]:


# Importing required libraries
from sklearn.datasets import fetch_20newsgroups

# Define the categories
categories = [
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc',
]

# Fetch the training data for the defined categories
train = fetch_20newsgroups(subset='train', categories=categories)

# Fetch the testing data for the defined categories
test = fetch_20newsgroups(subset='test', categories=categories)

# Print a sample of the training data
print(train.data[5])


# In[3]:


#importing necessary packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline
#creating a model based on multinomial Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
#Training the model with the train data
model.fit(train.data, train.target)
#Creating labels for the test data
labels = model.predict(test.data)


# In[8]:


#Creating the confussion matrix and heat map
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
            , xticklabels=train.target_names
            , yticklabels=train.target_names)
#plotting heat map of confussion matrix
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[11]:


#predicting category on new data based on trained model
def predict_category(s, train=train, model=model):
  pred= model.predict([s])
  return train.target_names[pred[0]]


# In[21]:


predict_category('The hunger games movie')


# In[ ]:





# In[ ]:





# In[ ]:




