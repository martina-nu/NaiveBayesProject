import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics


import unicodedata
import re

import pickle


url = 'https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv'
df = pd.read_csv(url)


def normalize_str(text_string):
    if text_string is not None:
        result = unicodedata.normalize("NFD",text_string).encode("ascii","ignore").decode()
    else:
        result = None
            
    return result


def preprocess(df):
    # Remove package name column
    df= df.drop('package_name', axis=1)
    
    # Convert text to lowercase and remove white spaces
    df['review'] = df['review'].str.strip().str.lower()
    
    #Remove symbols that are irrelevant
    df['review'] = df['review'].str.replace("!","")
    df['review'] = df['review'].str.replace(",","")
    df['review'] = df['review'].str.replace("&","")
    
    
    #Normalize text
    df['review'] = df['review'].str.normalize("NFKC")
    df['review']= df['review'].apply(normalize_str)
    
    #Remove extra letters in words (Loooove, Haaaate)
    df['review'] = df['review'].str.replace(r"([a-zA-Z])\1{2,}",r"\1",regex=True)
    
    
    return df


df = preprocess(df)

X = df['review']
y = df['polarity']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=15, stratify=y)

vec = CountVectorizer(stop_words='english')
X_train = vec.fit_transform(X_train).toarray()
X_test=vec.transform(X_test).toarray()


model = MultinomialNB()
model.fit(X_train, y_train)

filename = 'models/nb_model.sav'
pickle.dump(model, open(filename,'wb'))