#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:23:00 2020

@author: Jeet
"""

from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
import ast
from nltk import ngrams
import collections


def tokens_to_setence(df, column_name):
    #Convert tokens into sentences
    df[column_name] = df[column_name].apply(lambda x: ' '.join(list(ast.literal_eval(x))))
    return df

def generate_wordcloud(text) :
    
    # Create and generate a word cloud image:
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    
    # Display the generated image:
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# path to the preprocessed text file
file_path = r"text preprocessed file path"

# Read csv into pandas dataframe
filename = file_path+"filename.csv"
df = pd.read_csv(filename)

# Convert lemmatized tokens into sentences
df['lemmatized_setence'] = df['text_lemmatized']
df = tokens_to_setence(df, 'lemmatized_setence')

# Generate word cloud
text = ' '.join(text for text in df['lemmatized_setence'])
generate_wordcloud(text)

# Preare bigrams
df['text_bigrams'] = df['text_lemmatized'].apply(lambda x: ngrams(x, 2))
df['list_bigrams'] = df['text_bigrams'].apply(lambda x: list(x))

# Prepare trigrams 
df['text_trigrams'] = df['text_lemmatized'].apply(lambda x: ngrams(x, 3))
df['list_trigrams'] = df['text_trigrams'].apply(lambda x: list(x))

# Bigram list from all the data rows
list_bigrams = []
for bigrams in df['list_bigrams'] :
    list_bigrams.extend(bigrams)

# Frequency of each bigram
bigrams_freq = collections.Counter(list_bigrams)

# Ten most popular bigrams
bigrams_freq.most_common(10)

# Trigram list from all the data rows
list_trigrams = []
for trigrams in df['list_trigrams'] :
    list_trigrams.extend(trigrams)

#frequency of each trigram
trigrams_freq = collections.Counter(list_trigrams)

#ten most popular trigrams
trigrams_freq.most_common(10)
