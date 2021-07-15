#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)


# In[4]:


input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


# In[5]:


model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()


# In[6]:


bot_name = 'sam'


# In[8]:


def begin_chat_bot():
    print("Let's chat!, Type 'quit' to exit")
    while True:
        sentence = input('You: ')
        if (sentence == 'quit'):
            break

        sentence = tokenize(sentence)
        x = bag_of_words(sentence, all_words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x)

        output = model(x)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob > 0.75:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I don't understand...")


# In[9]:


begin_chat_bot()


# In[ ]:




