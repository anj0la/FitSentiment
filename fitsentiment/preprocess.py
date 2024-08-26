"""
File: preprocess.py

Author: Anjola Aina
Date Modified: August 26th, 2024

This file contains all the necessary functions used to preprocess the collected data.

Functions:
    fit(pd.DataFrame) -> tuple[list[list[int]], list[int]]: Preprocesses, tokenizes, labels, and encodes the text data.
    convert_to_csv(list[str] | list[int], list[str] | list[int], str) -> None: Converts the text data and their labels into a CSV file.
"""
import emoji
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from utils.lemmatize_text import lemmatize_text

# Normalize the text to remove variations of apostrophes and quotes
def tokenize_data(cleaned_data: list[str]) -> tuple[list[list[str]], dict[str, int]]:
    """
    Tokenizes the text data into words and creates a vocabulary dictionary.
        
    Args:
        cleaned_data (list[str]): The preprocessed data.
        
    Returns:
        tuple: A tuple containing:
            - list[list[str]]: The tokenized data.
            - dict[str, int]: The vocabulary dictionary mapping tokens to indices.
    """
    tokenized_data = []
    for sentence in cleaned_data:
        tokenized_data.extend([word_tokenize(term) for term in sent_tokenize(sentence)])
            
    all_tokens = [token for sentence in tokenized_data for token in sentence]
    vocab = {token: idx for idx, token in enumerate(set(all_tokens))}
            
    return tokenized_data, vocab

def _encode_token(tokenized_sentence: list[str], vocab: dict) -> list[int]:
    """
    Encodes a token vector into a list of integers using the vocabulary dictionary.
        
    Args:
        token_vector (list[str]): A tokenized sentence.
        vocab (dict): The vocabulary dictionary.
        
    Returns:
        list[int]: The encoded token vector.
    """
    return [vocab[token] for token in tokenized_sentence]
        
def encode_tokens(tokenized_data: list[list[str]], vocab: dict) -> list[list[int]]:
    """
    Encodes multiple token vectors into lists of integers using the vocabulary dictionary.
        
    Args:
        token_vectors (list[list[str]]): The tokenized text data.
        vocab (dict): The vocabulary dictionary.
        
    Returns:
        list[list[int]]: The encoded token vectors.
    """
    return [_encode_token(tokenized_sentence, vocab) for tokenized_sentence in tokenized_data]
 
def preprocess(file_path: str) -> list[str]:
    """
    This function preprocesses the text data, returning the processed data.
     
    It preprocesses the text data by converting the text to lowercase and emojis to text, removing punctuation, special characters,
    links, email addresses and applying lemmatization.
        
    Args:
        file_path (str): The file path containing the text data.
        
    Returns:
        list[str]: The cleaned data.
    """
    df = pd.read_csv(file_path)
    data = df['text']
        
    # convert the text to lowercase
    data = data.str.lower()
    
    # remove punctuation and special characters
    data = data.replace(r'[.,;:!\?"\'`]', '', regex=True)
    data = data.replace(r'[@#\$%^&*\(\)\\/\+\-_=\\[\]\{\}<>]', '', regex=True)
    data = data.replace(r'[‘’“”]', '', regex=True)
    
    # convert emojis to text
    data = data.apply(lambda x: emoji.demojize(x))
    data = data.replace(r':', '', regex=True)
        
    # remove links and email addresses
    data = data.replace(r'http\S+|www\.\S+', '', regex=True)
    data = data.replace(r'\w+@\w+\.com', '', regex=True)
        
    # remove stop words and apply lemmatization
    stop_words = set(stopwords.words('english'))
    data = data.apply(lambda sentence: ' '.join(word for word in sentence.split() if word not in stop_words))
    # data = data.apply(lambda sentence: lemmatize_text(sentence))
            
    return data.values  

# Testing purposes
preprocessed_data = preprocess('data/test_corpus.csv')
print(preprocessed_data[:5])
tokenized_data, vocab = tokenize_data(cleaned_data=preprocessed_data)
print('\n============================== TOKENIZED DATA ==============================\n')
print(tokenized_data[:5])
print(vocab)
