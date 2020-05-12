#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:34:40 2020

@author: Jeet
"""

import pandas as pd
import nltk
import string
import re
import ast


#function to remove punctuation
def remove_punctuation(text) :
    #remove all punctuations
    text_nopunct = "".join(char for char in text if char not in string.punctuation)
    return text_nopunct

#function to tokenize worfs
def tokenize(text) :
    #W+ means either alphanumeric or a dash(-)
    tokens = re.split('\W+', text)
    return tokens
    
def remove_stopwords(tokenized_list) :
    stopwords = nltk.corpus.stopwords.words('english')
    #remove english stopwords
    text_no_stopwords = [word for word in tokenized_list if word not in stopwords]
    return text_no_stopwords

def stemmer(tokenized_list) :
    ps = nltk.PorterStemmer()
    stemmed_text = [ps.stem(word) for word in tokenized_list]
    return stemmed_text

def lemmatize(tokenized_list) :
    wn = nltk.WordNetLemmatizer()
    lemmatized_text = [wn.lemmatize(word) for word in tokenized_list]
    return lemmatized_text

def tokens_to_setence(df, column_name):
    #Convert tokens into sentences
    df[column_name] = df[column_name].apply(lambda x: ' '.join(list(ast.literal_eval(x))))
    return df

# Path to the input file
file_path = r"file_path"

# Read csv into pandas dataframe
filename = file_path+"file_name.csv"
df = pd.read_csv(filename, engine='python')


#convert text to lower
df['text_lower'] = df['text_column'].apply(lambda x: x.lower())

#remove punctuation from reviews
df['text_nopunct'] = df['text_lower'].apply(lambda x: remove_punctuation(x))

#tokenize reviews
df['text_toeknized'] = df['text_nopunct'].apply(lambda x: tokenize(x))

#remove stopwords from reviews
df['text_no_stopwords'] = df['text_toeknized'].apply(lambda x: remove_stopwords(x))

#stem reviews
df['text_stemmed'] = df['text_no_stopwords'].apply(lambda x: stemmer(x))

#lemmatize reviews
df['text_lemmatized'] = df['text_no_stopwords'].apply(lambda x: lemmatize(x))

# Convert lemmatized tokens into sentences
#df['lemmatized_setence'] = df['text_lemmatized']
#df = tokens_to_setence(df, 'lemmatized_setence')

# Save preprocessed text to csv
file_path_preprocessed = file_path+"file_name_preprocessed.csv"
df.to_csv(file_path_preprocessed)

