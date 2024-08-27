"""
File: feature_extraction.py

Author: Anjola Aina
Date Modified: August 26th, 2024

This file contains all the necessary functions used to extract features from the collected data.

Functions:
    TODO: Populate this section with functions.
"""
from constants.constants import FEATURES
from fuzzywuzzy import fuzz
import pandas as pd
import spacy
import nltk

def extract_features(file_path):
    df = pd.read_csv(file_path)
    data = df['text']

    extracted_features = []
    
    for sentence in data.array:
        sentence_matched = False
        
        for feature in FEATURES:
            if feature in sentence:
                extracted_features.append((feature, sentence))
                sentence_matched = True
                break  # exit the loop once a feature is matched

        if not sentence_matched:
            for word in sentence.split():
                if any(fuzz.ratio(feature, word) > 80 for feature in FEATURES):
                    extracted_features.append((feature, sentence))
                    sentence_matched = True
                    break
        
        if not sentence_matched:
            extracted_features.append((None, sentence))

    return extracted_features

# Testing
extracted_features = extract_features('data/test_corpus.csv')
dataframe = pd.read_csv('data/test_corpus.csv')
text = ''.join(dataframe['text'].values.tolist())
nlp = spacy.load('en_core_web_sm')

doc = nlp(text)
print(doc)

entities = []
labels = []

for ent in doc.ents:
	entities.append(ent)
	labels.append(ent.label_)

df = pd.DataFrame({'Entities': entities, 'Labels': labels})
df
print('\n============================== ENTITY LABEL PAIRS (SPACY) ==============================\n')

for ent, label in zip(entities, labels):
    print(ent, label)
    
print('\n============================== ENTITY LABEL PAIRS (NLTK) ==============================\n')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

tokenize = word_tokenize(text)

stopwords_tokenize =  [word for word in tokenize if word not in stop_words]
pos_tags = pos_tag(stopwords_tokenize)

#for pos_tag in pos_tags:
 #   print(pos_tag)

print('\n============================== EXTRACTED FEATURES ==============================\n')
print(extracted_features[:2])