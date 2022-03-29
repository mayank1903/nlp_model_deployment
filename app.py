#!/usr/bin/env python
# coding: utf-8

# In[20]:

from collections.abc import Mapping
import pickle
from flask import Flask, render_template, url_for,request
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB


# In[22]:


lemmatizer = WordNetLemmatizer()


# In[23]:


def clean_question_final(text):
    review = re.sub(r'<.*?>', '', text)
    review = re.sub('&', ' ', review)
    review = re.sub('nbsp;', ' ', review)
    review = re.sub('\s+',' ',review)
    review = re.sub('[^a-zA-Z0-9 ]', '', review)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(words) for words in review if words not in set(stopwords.words('english'))]
    review = [items for items in review if len(items) > 1]
    return ' '.join(review)


# In[39]:


filename = 'nlp_model.pkl'
tfidf = pickle.load(open('transform.pkl', 'rb'))
model = pickle.load(open(filename, 'rb'))


# In[ ]:


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        message = clean_question_final(message)
        data = [message]
        vect = tfidf.transform(data).toarray()
        my_prediction = model.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == "__main__":
    #app.debug = True
    app.run(debug = True)

