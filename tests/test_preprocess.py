"""
File: test_preprocess.py

Author: Anjola Aina
Date Modified: August 26th, 2024

Description:
    This file contains all the necessary functions to test the preprocessing data functionality, which aims to clean the corpus
    by removing symbols and text.
"""
from fitsentiment.preprocess import tokenize_data, preprocess
import os
import tempfile
import pandas as pd
import unittest

class TestPreprocess(unittest.TestCase):

    def setUp(self):
        # setting up the temporary csv file
        self.test_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        self.test_csv_path = self.test_csv.name
        data = pd.DataFrame({
            'text': [
                'I like to use loseit! because it has a barcode scanner.',
                'MyFitnessPal allows me to track my calories which is awesome',
                'Use macrofactor for tracking your macros, super useful'
            ]
        })
        data.to_csv(self.test_csv_path, index=False)
        self.test_csv.close()  # properly closing the file
        
       # expected results from other functions
        self.expected_cleaned_data = [
            'like use loseit barcode scanner',
            'myfitnesspal allows track calories awesome',
            'use macrofactor tracking macros super useful'
        ]
        self.expected_tokenized_data = [
            ['like', 'use', 'loseit', 'barcode', 'scanner'],
            ['myfitnesspal', 'allows', 'track', 'calories', 'awesome'],
            ['use', 'macrofactor', 'tracking', 'macros', 'super', 'useful']
        ]
        self.expected_vocab_keys = {
            'like', 'use', 'loseit', 'barcode', 'scanner', 
            'myfitnesspal', 'allows', 'track', 'calories', 'awesome', 
            'macrofactor', 'tracking', 'macros', 'super', 'useful'
        }
        
    def tearDown(self):
        # removes the temporary file
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
            
    def test_preprocess(self):
        cleaned_data = preprocess(self.test_csv_path).tolist()
        
        # ensure cleaned data is equal to the expected cleaned data
        self.assertListEqual(cleaned_data, self.expected_cleaned_data, f'Data not fully cleaned.\nExpected data: {self.expected_cleaned_data}\nActual data: {cleaned_data}')
        
    def test_tokenize_data(self):
        tokenized_data, vocab = tokenize_data(self.expected_cleaned_data)
        
        # ensure tokenized data is equal to expected 
        self.assertListEqual(tokenized_data, self.expected_tokenized_data)
        
        # check that the dictionary has the expected number of keys
        self.assertEqual(len(vocab), len(self.expected_vocab_keys), 'Dictionary length is incorrect.')
        
        # check that all keys in the expected vocab exist in the actual vocab
        for key in self.expected_vocab_keys:
                self.assertIn(key, vocab, f'Missing key: {key}')
        
        # check conditions on values (i.e., an integer value greater than or equal to 0)
        for key in vocab:
            self.assertIsInstance(vocab[key], int, f'Value for key {key} is not an integer.')
            self.assertGreaterEqual(vocab[key], 0, f'Value for key {key} should be greater than (or equal to) 0.')

if __name__ == '__main__':
    unittest.main()