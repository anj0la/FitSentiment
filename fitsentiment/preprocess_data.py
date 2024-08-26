"""
File: preprocess_data.py

Author: Anjola Aina
Date Modified: August 26th, 2024

This file contains all the necessary functions used to preprocess the collected data.

Functions:
    fit(pd.DataFrame) -> tuple[list[list[int]], list[int]]: Preprocesses, tokenizes, labels, and encodes the text data.
    convert_to_csv(list[str] | list[int], list[str] | list[int], str) -> None: Converts the text data and their labels into a CSV file.
"""
import csv
import emoji
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import sent_tokenize, word_tokenize
from utils.lemmatize_text import lemmatize_text

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
        """
        Preprocesses the text data by converting to lowercase, removing punctuation and special characters,
        converting emojis to text, removing links and email addresses, and applying lemmatization.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the text data.
        
        Returns:
            list[str]: The preprocessed text data.
        """
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
        """
        Tokenizes the text data into words and creates a vocabulary dictionary.
        
        Args:
            data (list[str]): The preprocessed text data.
        
        Returns:
            tuple: A tuple containing:
                - list[list[str]]: The tokenized text data.
                - dict: The vocabulary dictionary mapping tokens to indices.
        """
        token_vectors = []
        for sentence in data:
            token_vectors.extend([word_tokenize(term) for term in sent_tokenize(sentence)])
            
        all_tokens = [token for vector in token_vectors for token in vector]
        self.vocab = {token: idx for idx, token in enumerate(set(all_tokens))}
            
        return token_vectors        

    def _encode_token(self, token_vector: list[str], vocab: dict) -> list[int]:
        """
        Encodes a token vector into a list of integers using the vocabulary dictionary.
        
        Args:
            token_vector (list[str]): A tokenized sentence.
            vocab (dict): The vocabulary dictionary.
        
        Returns:
            list[int]: The encoded token vector.
        """
        return [vocab[token] for token in token_vector]
        
    def _encode_tokens(self, token_vectors: list[list[str]], vocab: dict) -> list[list[int]]:
        """
        Encodes multiple token vectors into lists of integers using the vocabulary dictionary.
        
        Args:
            token_vectors (list[list[str]]): The tokenized text data.
            vocab (dict): The vocabulary dictionary.
        
        Returns:
            list[list[int]]: The encoded token vectors.
        """
        return [self._encode_token(vector, vocab) for vector in token_vectors]
    
    def _encode_labels(self, labels: list[str]) -> list[int]:
        """
        Encodes labels into integer values using LabelEncoder.
        
        Args:
            labels (list[str]): The labels for each token vector.
        
        Returns:
            list[int]: The encoded labels.
        """
        # use label_encoder.inverse_transform to decode the labels
        return LabelEncoder().fit_transform(labels)
    
    def convert_to_csv(self, text: list[str] | list[int], labels: list[str] | list[int], file_path: str) -> None:
        """
        Converts the text data and their labels into a CSV file.
        
        Args:
            text (list[str] | list[int]): The text data.
            labels (list[str] | list[int]): The labels for each piece of text.
            file_path (str): The path to save the CSV file.
        """
        fields = ['text', 'label']
        rows = []
        for sentence, label in zip(text, labels):
            rows.append({'text': sentence, 'label': label})
        with open(file_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            csv_file.close()  

    def fit(self, df: pd.DataFrame) -> tuple[list[list[int]], list[int]]:
        """
        Preprocesses, tokenizes, labels, and encodes the text data.
        
        Args:
            df (pd.DataFrame): The input DataFrame containing the text data.
        
        Returns:
            tuple: A tuple containing:
                - list[list[int]]: The encoded token vectors.
                - list[int]: The encoded labels.
        """
        # processing, tokenizing and labelling the data
        preprocessed_data = self._preprocess_data(df=df)
        tokenized_data = self._tokenize_data(data=preprocessed_data)
        # labels = self._label_data(token_vectors=tokenized_data)
        
        # encoding the token vectors and labels
        #encoded_vectors = self._encode_tokens(token_vectors=tokenized_data, vocab=self.vocab)
        #encoded_labels = self._encode_labels(labels=labels)
        return tokenized_data # placeholder for testing
        
        # return encoded_vectors, encoded_labels
    
# Testing purposes
df = pd.read_csv('data/corpus.csv')

# preprocessing the corpus and encoding the text and labels
text_pipeline = TextPipeline()
tokenized_data = text_pipeline.fit(df)
print(tokenized_data)