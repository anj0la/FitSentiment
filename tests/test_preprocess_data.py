"""
File: test_preprocess_data.py

Author: Anjola Aina
Date Modified: June 3rd, 2024

Description:
    This file contains all the necessary functions to test the preprocessing data functionality, which aims to "clean" the corpus
    by removing irrelevant symbols and text that may affect the classification results, and then convert it into input that is
    compatible for a machine learning model.
    
    Note: The vocabulary is created simply by iterating through each token in the token vectors instead of using the TdifVectorizer for simplicity reasons.
    
    Currently, the test file has a coverage of ____%.
"""

from fitsentiment.preprocess_data import TextPipeline
from constants.constants import WORKOUT_CLASSES

def test_preprocess_data():
    pipeline = TextPipeline()
    corpus = [
        'I want to use this so bad but I feel it should have a rest day Thursday then do the rest, do you think that will help you more?',
        'You absolutely must incorporate squats into your leg workout as well as deadlifts (either also on leg day or on back day). Those are two of the three most important and effective lifts that hit well beyond your legs.'
    ]
    expected_preprocessed_data = [
        'want use bad feel rest day thursday rest think help',
        'absolutely must incorporate squat leg workout well deadlifts either also leg day back day two three important effective lift hit well beyond leg'
    ]
    preprocessed_data = pipeline._preprocess_data(corpus)
    assert all(data == expected_data for data, expected_data in zip(preprocessed_data, expected_preprocessed_data))


def test_tokenize_data():
    pipeline = TextPipeline()
    corpus = [
        'want use bad feel rest day thursday rest think help',
        'absolutely must incorporate squat leg workout well deadlifts either also leg day back day two three important effective lift hit well beyond leg'
    ]
    tokenized_data, vocab = pipeline._tokenize_data(corpus)
    assert isinstance(tokenized_data, list)
    assert isinstance(vocab, dict)
    assert all(isinstance(sentence, list) for sentence in tokenized_data)

def test_label_data():
    pipeline = TextPipeline()
    tokenized_data = [
        ['want', 'use', 'bad', 'feel', 'rest', 'day', 'thursday', 'rest', 'think', 'help'], 
        ['absolutely', 'must', 'incorporate', 'squat', 'leg', 'workout', 'well', 'deadlifts', 'either', 'also', 'leg', 'day', 'back', 'day', 'two', 'three', 'important', 'effective', 'lift', 'hit', 'well', 'beyond', 'leg']
    ]
    labels = pipeline._label_data(tokenized_data)
    assert isinstance(labels, list)
    assert all(label in WORKOUT_CLASSES for label in labels)

def test_extract_features():
    pipeline = TextPipeline()
    token_vectors = [
        ['want', 'use', 'bad', 'feel', 'rest', 'day', 'thursday', 'rest', 'think', 'help'], 
        ['absolutely', 'must', 'incorporate', 'squat', 'leg', 'workout', 'well', 'deadlifts', 'either', 'also', 'leg', 'day', 'back', 'day', 'two', 'three', 'important', 'effective', 'lift', 'hit', 'well', 'beyond', 'leg']
    ]
    labels = ['general fitness', 'lower body']
    features = pipeline._extract_features(token_vectors, labels)
    assert isinstance(features, list)
    assert all(isinstance(feature, tuple) for feature in features)

def test_encode_token():
    pipeline = TextPipeline()
    token_vector = ['want', 'use', 'bad', 'feel', 'rest', 'day', 'thursday', 'rest', 'think', 'help']
    vocab = {word: i for i, word in enumerate(set(token_vector))}
    encoded_vector = pipeline._encode_token(token_vector, vocab)
    assert isinstance(encoded_vector, list)
    assert all(isinstance(token, int) for token in encoded_vector)

def test_encode_tokens():
    pipeline = TextPipeline()
    token_vectors = [
        ['want', 'use', 'bad', 'feel', 'rest', 'day', 'thursday', 'rest', 'think', 'help'], 
        ['absolutely', 'must', 'incorporate', 'squat', 'leg', 'workout', 'well', 'deadlifts', 'either', 'also', 'leg', 'day', 'back', 'day', 'two', 'three', 'important', 'effective', 'lift', 'hit', 'well', 'beyond', 'leg']
    ]
    vocab = {word: i for i, word in enumerate(set(word for sentence in token_vectors for word in sentence))}
    encoded_vectors = pipeline._encode_tokens(token_vectors, vocab)
    assert isinstance(encoded_vectors, list)
    assert all(isinstance(vector, list) for vector in encoded_vectors)

# Test for _encode_labels method
def test_encode_labels():
    pipeline = TextPipeline()
    labels = ['general fitness', 'lower body']
    encoded_labels = pipeline._encode_labels(labels)
    expected_encoded_labels = [0, 1]
    assert all(encoded_label == expected_label for encoded_label, expected_label in zip(encoded_labels, expected_encoded_labels))

# Test for _pad_vectors method
def test_pad_vectors():
    pipeline = TextPipeline()
    encoded_vectors = [[1, 2, 3], [4, 5, 6, 7, 8]]
    padded_vectors = pipeline._pad_vectors(encoded_vectors)
    max_length = max(len(vector) for vector in encoded_vectors)
    assert all(len(vector) == max_length for vector in padded_vectors)

# Test for fit method
def test_fit():
    pipeline = TextPipeline()
    corpus = [
        'I want to use this so bad but I feel it should have a rest day Thursday then do the rest, do you think that will help you more?',
        'You absolutely must incorporate squats into your leg workout as well as deadlifts (either also on leg day or on back day). Those are two of the three most important and effective lifts that hit well beyond your legs.'
    ]
    features, vocab = pipeline.fit(corpus)
    assert isinstance(features, list)
    assert isinstance(vocab, dict)
    assert all(isinstance(feature, tuple) for feature in features)
