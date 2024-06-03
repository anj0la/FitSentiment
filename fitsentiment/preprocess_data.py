"""
File: preprocess_data.py

Author: Anjola Aina
Date Modified: June 2nd, 2024

Description:
    This file contains all the necessary functions used to preprocess relevant data about fitness.
    There is one public function, fit, which extracts features and the vocabulary from the corpus.

Functions:
    fit(corpus: list[str]) -> pd.Dataframe: Removes punctuation and special characters, tokenizes data, and extracs features from the corpus.
"""
import torch
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

    def _preprocess_data(self, corpus: list[str]) -> list[str]:
        df = pd.DataFrame(data=corpus, columns=['text'], dtype='string')
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
        self.vectorizer.fit(data)
        vocab = self.vectorizer.vocabulary_
        
        token_vectors = []
        for sentence in data:
            token_vectors.extend([word_tokenize(term) for term in sent_tokenize(sentence)])
            
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
        return LabelEncoder().fit_transform(labels)
    
    def _pad_vectors(self, encoded_vectors: list[list]) -> list[list]:
        max_length = max(len(vector) for vector in encoded_vectors)
        padded_vectors = [vector + [0] * (max_length - len(vector)) for vector in encoded_vectors]
        return padded_vectors


    def fit(self, corpus: list[str]) -> tuple[list[tuple], dict]:
        # processing, tokenizing and labelling the data
        preprocessed_data = self._preprocess_data(corpus=corpus)
        tokenized_data, vocab = self._tokenize_data(data=preprocessed_data)
        labels = self._label_data(token_vectors=tokenized_data)
        
        # encoding the token vectors and labels
        encoded_vectors = self._encode_tokens(token_vectors=tokenized_data, vocab=vocab)
        encoded_labels = self._encode_labels(labels=labels)
        
        # padding to ensure inputs to ml are the same length
        padded_vectors = self._pad_vectors(encoded_vectors=encoded_vectors)
        
        # converting to tensors (so input is compatiable)
        input_tensors = torch.tensor(padded_vectors, dtype=torch.float32)
        label_tensors = torch.tensor(encoded_labels, dtype=torch.long)
        
        # create (feature, target) pairs
        features = self._extract_features(token_vectors=input_tensors, labels=label_tensors)
        return features, vocab
    
# Usage 
text_pipeline = TextPipeline()

TEST_CORPUS: tuple[str] = ('I want to use this so bad but I feel it should have a rest day Thursday then do the rest, do you think that will help you more?', 'You absolutely must incorporate squats into your leg workout as well as deadlifts (either also on leg day or on back day). Those are two of the three most important and effective lifts that hit well beyond your legs')
feat, voc = text_pipeline.fit(TEST_CORPUS)
print('features: ', feat)
print()
print('vocab: ', voc)