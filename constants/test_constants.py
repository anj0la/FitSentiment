"""
File: test_constants.py

Author: Anjola Aina
Date Modified: May 29th, 2024

Description:
    This file contains all the necessary constant classes that are used to test the model. They are distingushed differently from the constants.py file,
    even though they contain similar classes and attributes to "mock" the information that should be returned from the tested functions.

Classes:
    REDDIT_SCRAPER_RESPONSES: All constants related to the reponses returned from scraping relevant information from Reddit.
"""
from dataclasses import dataclass

TEST_CORPUS: tuple[str] = ('I want to use this so bad but I feel it should have a rest day Thursday then do the rest, do you think that will help you more?', 'You absolutely must incorporate squats into your leg workout as well as deadlifts (either also on leg day or on back day). Those are two of the three most important and effective lifts that hit well beyond your legs')

@dataclass()
class REDDIT_SCRAPER_RESPONSES:
    """
    This test data class contains all of the constants related to the Reddit scraper. The attributes are immutable,
    so that they cannot be changed after creation.
    """
    TEST_CORPUS: tuple[str] = ('I want to use this so bad but I feel it should have a rest day Thursday then do the rest, do you think that will help you more?', 'You absolutely must incorporate squats into your leg workout as well as deadlifts (either also on leg day or on back day). Those are two of the three most important and effective lifts that hit well beyond your legs')
    TEST_COMMENT: str = 'Thanks for the tip, I switched triceps to chest day and put biceps with shoulder.\n\nWhat day are you working traps?'
    TEST_SPLIT_COMMENT: tuple[str] = ('Thanks for the tip, I switched triceps to chest day and put biceps with shoulder.', 'What day are you working traps?')
    LEN_CORPUS: int = 2
    LEN_COMMENT: int = 2
    
