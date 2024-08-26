"""
File: feature_context.py

Author: Anjola Aina
Date Modified: August 24th, 2024

This file contains the data class used to store the text, app tag, and features.
The dataclass is initially created with the text (other values are set to None).

"""
from dataclasses import dataclass

@dataclass
class FeatureContext:
    def __init__(self, text, app_tag = None, features = None):
        self.text = text
        self.app_tag = app_tag
        self.features = features