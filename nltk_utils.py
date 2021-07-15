#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
import numpy as np


# In[2]:


stemmer = PorterStemmer()


# In[3]:


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# In[4]:


def stem(word):
    return stemmer.stem(word.lower())


# In[10]:


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    
    for i, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[i] = 1
    
    return bag

