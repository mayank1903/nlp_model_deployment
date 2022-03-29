#!/usr/bin/env python
# coding: utf-8

# In[20]:

from collections.abc import Mapping
import pickle
from flask import Flask, render_template, url_for,request
import pandas as pd
import numpy as np
import nltk
#from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import joblib
#from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB


# In[22]:


lemmatizer = WordNetLemmatizer()

stopwords = ['just',
 'll',
 "should've",
 'from',
 'than',
 "you'd",
 'where',
 'too',
 'below',
 'at',
 'hadn',
 'mightn',
 'ourselves',
 'now',
 'has',
 'until',
 'there',
 'our',
 'to',
 "wasn't",
 'we',
 'itself',
 'when',
 'and',
 'shan',
 'me',
 'have',
 "hadn't",
 'shouldn',
 're',
 'more',
 'theirs',
 'their',
 "shouldn't",
 'against',
 'its',
 'won',
 'for',
 'his',
 'in',
 'will',
 'of',
 'by',
 'again',
 'or',
 "didn't",
 'before',
 'does',
 'here',
 'am',
 'he',
 'wasn',
 'your',
 "mightn't",
 "mustn't",
 'through',
 'mustn',
 'which',
 'yours',
 "won't",
 "you'll",
 'themselves',
 'during',
 "you're",
 "you've",
 'that',
 'don',
 'you',
 'this',
 'needn',
 'i',
 'out',
 'being',
 'having',
 'was',
 'any',
 'what',
 "isn't",
 'ours',
 'did',
 'into',
 'weren',
 'it',
 'then',
 'each',
 "couldn't",
 'all',
 'other',
 'nor',
 'over',
 'most',
 'are',
 'were',
 'both',
 "she's",
 'some',
 'a',
 'yourselves',
 'hers',
 'as',
 've',
 'she',
 'off',
 'why',
 'doing',
 's',
 'whom',
 'had',
 'so',
 'm',
 'they',
 'himself',
 'ma',
 'these',
 'very',
 'couldn',
 "shan't",
 'been',
 'while',
 'own',
 'between',
 'can',
 'd',
 'him',
 "it's",
 'isn',
 'how',
 "don't",
 'the',
 'under',
 'not',
 'those',
 'only',
 'ain',
 'once',
 'do',
 'after',
 'hasn',
 'y',
 "weren't",
 "aren't",
 'wouldn',
 'up',
 'few',
 "that'll",
 'myself',
 't',
 'is',
 'yourself',
 'same',
 'about',
 'doesn',
 'be',
 'an',
 'down',
 'if',
 'above',
 'no',
 'herself',
 'should',
 "haven't",
 'who',
 'them',
 'further',
 'haven',
 'my',
 'on',
 'such',
 'with',
 "needn't",
 "doesn't",
 'her',
 'aren',
 "hasn't",
 'o',
 'didn',
 "wouldn't",
 'because',
 'but']


# In[23]:


def clean_question_final(text):
    review = re.sub(r'<.*?>', '', text)
    review = re.sub('&', ' ', review)
    review = re.sub('nbsp;', ' ', review)
    review = re.sub('\s+',' ',review)
    review = re.sub('[^a-zA-Z0-9 ]', '', review)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(words) for words in review if words not in stopwords]
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

