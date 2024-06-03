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
import emoji
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from utils.lemmatize_text import lemmatize_text
from constants.constants import WORKOUT_CLASSES, UPPER_BODY_PARTS, LOWER_BODY_PARTS, CORE_PARTS, FULL_BODY_KEYWORDS, UPPER_LOWER_KEYWORDS, PUSH_PULL_LEGS_KEYWORDS

class TextPipeline:
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
            
    def _label_data(self, token_vectors):
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

    def _extract_features(self, token_vectors: list[str]) -> pd.DataFrame:
        features = []
        labels = self._label_data(token_vectors)
        for token_vector, label in zip(token_vectors, labels):
            features.append((token_vector, label))
        return features

    def fit(self, corpus: list[str]) -> pd.DataFrame:
        preprocessed_data = self._preprocess_data(corpus)
        
        print("\n================= after preprocessing =================\n")
        print(preprocessed_data)
        print()

        tokenized_data, vocab = self._tokenize_data(preprocessed_data)

        print("================= after tokenizing =================\n")
        print("tokenized data: ", tokenized_data)
        print()
        print("vocabulary: ", vocab)
        print()

        features = self._extract_features(preprocessed_data)

        print("================= after feature extraction =================\n")
        print(features)
        print()

        return features, vocab
    


# Usage example
TEST_CORPUS = ('😃 I wan\'t to use this so "bad" but I feel it should have a rest day Thursday then do the rest, do you think that will help you more?', 'You absolutely must incorporate squats into your leg workout as well as deadlifts (either also on leg day or on back day). Those are two of the three most important and effective lifts that hit well beyond your legs')
text_pipeline = TextPipeline()
features, vocab = text_pipeline.fit(TEST_CORPUS)

print('================= features  =================')

print(features)
print()

print('================= vocab =================')

print(vocab)
print()

