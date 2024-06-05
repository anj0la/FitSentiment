"""
File: preprocess_data.py

Author: Anjola Aina
Date Modified: June 3rd, 2024

Description:
    This file contains all the necessary functions used to preprocess relevant data about fitness.
    There is one public function, fit, which extracts features and the vocabulary from the corpus.

Functions:
    fit(corpus: list[str]) -> pd.Dataframe: Removes punctuation and special characters, tokenizes data, and extracs features from the corpus.
"""

import csv
import emoji
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import sent_tokenize, word_tokenize
from utils.lemmatize_text import lemmatize_text
from constants.constants import WORKOUT_CLASSES, UPPER_BODY_PARTS, LOWER_BODY_PARTS, CORE_PARTS, FULL_BODY_KEYWORDS, UPPER_LOWER_KEYWORDS, PUSH_PULL_LEGS_KEYWORDS

class TextPipeline:
    """
    This class is used to create a text pipeline to transform the corpus into data that can be used for an input to a machine learning model. 
    
    Attributes:
        stop_words (set[str]): The set of stop words to remove from the corpus.
        vectorizer (TfidfVectorizer: The vectorizer, used to create the vocabulary from the corpus.
        classes (tuple[str]): The classes that each submission will belong to.
        
    Public Functions:
        fit(self, str): -> DataFrame
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self.classes = WORKOUT_CLASSES
        self.vocab = None

    def _preprocess_data(self, df: pd.DataFrame) -> list[str]:
        data = df['text']
        
        # convert the text to lowercase
        data = data.str.lower()

        # remove punctuation and special characters
        data = data.replace(r'[.,;:!\?"\'`]', '', regex=True)
        data = data.replace(r'[@#\$%^&*\(\)\\/\+\-_=\\[\]\{\}<>]', '', regex=True)
        
        # convert emojis to text
        data = data.apply(lambda x: emoji.demojize(x))
        data = data.replace(r':', '', regex=True)
        
        # remove links and email addresses
        data = data.replace(r'http\S+|www\.\S+', '', regex=True)
        data = data.replace(r'\w+@\w+\.com', '', regex=True)
        
        # remove stop words and apply lemmatization
        data = data.apply(lambda sentence: ' '.join(word for word in sentence.split() if word not in self.stop_words))
        data = data.apply(lambda sentence: lemmatize_text(sentence))
        
        return data.values

    def _tokenize_data(self, data: list[str]) -> tuple[list[list[str]], dict]:
        token_vectors = []
        for sentence in data:
            token_vectors.extend([word_tokenize(term) for term in sent_tokenize(sentence)])
            
        all_tokens = [token for vector in token_vectors for token in vector]
        vocab = {token: idx for idx, token in enumerate(set(all_tokens))}
            
        return token_vectors, vocab        
            
    def _label_data(self, token_vectors: list[list[str]]) -> list[str]:
        classes = []
        for token_vector in token_vectors:
        # initialize flags for each type of workout
            has_lower_body = any(part in token_vector for part in LOWER_BODY_PARTS)
            has_upper_body = any(part in token_vector for part in UPPER_BODY_PARTS)
            has_core = any(part == token_vector for part in CORE_PARTS) # ignores words like absolutely which has 'abs' in it
            has_full_body_keywords = any(keyword in token_vector for keyword in FULL_BODY_KEYWORDS)
            has_upper_lower_keywords = any(keyword in token_vector for keyword in UPPER_LOWER_KEYWORDS)
            has_push_pull_legs_keywords = any(keyword in token_vector for keyword in PUSH_PULL_LEGS_KEYWORDS)
            
            # case one: full body (contains at least one lower and upper body part and core/full body keywords)
            if (has_lower_body and has_upper_body and has_core) or has_full_body_keywords:
                classes.append(WORKOUT_CLASSES[0])  # class 0 = full body
            # case two: upper/lower split
            elif has_upper_lower_keywords or (has_lower_body and has_upper_body):
                classes.append(WORKOUT_CLASSES[1])  # class 1 = upper lower split
            # case three: push/pull/legs
            elif has_push_pull_legs_keywords: # change
                classes.append(WORKOUT_CLASSES[2])  # class 2 = push pull legs
            # case four: lower body
            elif has_lower_body and not has_upper_body:
                classes.append(WORKOUT_CLASSES[3])  # class 3 = lower body
            # case five: upper body
            elif has_upper_body and not has_lower_body:
                classes.append(WORKOUT_CLASSES[4])  # class 4 = upper body
            else:
                classes.append(WORKOUT_CLASSES[5])  # if none of the above cases match, assume it is general fitness
        return classes

    def _extract_features(self, token_vectors: list[list[str]], labels: list[str]) -> list[tuple]:
        features = []
        for token_vector, label in zip(token_vectors, labels):
            features.append((token_vector, label))
        return features
    
    def _encode_token(self, token_vector: list[str], vocab: dict) -> list:
            return [vocab[token] for token in token_vector]
        
    def _encode_tokens(self, token_vectors: list[list[str]], vocab: dict) -> list[list]:
        return [self._encode_token(vector, vocab) for vector in token_vectors]
    
    def _encode_labels(self, labels: list[str]) -> list:
        # use label_encoder.inverse_transform to decode the labels
        return LabelEncoder().fit_transform(labels)
    
    def convert_to_csv(self, text, labels, file_path):
        fields = ['text', 'label']
        rows = []
        for sentence, label in zip(text, labels):
            rows.append({'text': sentence, 'label': label})
        with open(file_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            csv_file.close()  

    def fit(self, df: pd.DataFrame) -> tuple[list[list[str]], dict]:
        # processing, tokenizing and labelling the data
        preprocessed_data = self._preprocess_data(df=df)
        tokenized_data, vocab = self._tokenize_data(data=preprocessed_data)
        labels = self._label_data(token_vectors=tokenized_data)
        
        # encoding the token vectors and labels
        encoded_vectors = self._encode_tokens(token_vectors=tokenized_data, vocab=vocab)
        encoded_labels = self._encode_labels(labels=labels)
        
        return encoded_vectors, encoded_labels, vocab
    
    
