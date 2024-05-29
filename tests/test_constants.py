"""
File: test_constants.py

Author: Anjola Aina
Date Modified: May 29th, 2024

Description:

This file contains all the necessary constant classes that are used to test the model. They are distingushed differently from the constants.py file,
even though they contain similar classes and attributes to "mock" the information that should be returned from the tested functions.

Classes:
    TEST_REDDIT_SCRAPER_CONSTANTS: All constants related to scraping relevant information from reddit.

Sources: N/A
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class REDDIT_SCRAPER_RESPONSE:
    """
    This test data class contains all of the constants related to the reddit scraper. The class and its attributes are immutable,
    so that they cannot be changed after creation.
    """
    TEST_CORPUS: tuple[str] = ('I want to use this so bad but I feel it should have a rest day Thursday then do the rest, do you think that will help you more?', 'You absolutely must incorporate squats into your leg workout as well as deadlifts (either also on leg day or on back day). Those are two of the three most important and effective lifts that hit well beyond your legs', 'Taking rest on friday, after one complete cycle.', "I can't do squats and deadlifts due to lower back injury but i will add them once i feel better. \nThanks man", 'But what if your other muscles are actually sore due to progressive overload? Assuming that’s what you’re doing and you are trying hard to build up that muscle!', 'Apologies - makes total sense then. Be safe. Maybe try hack squats before you even do start squats when your back gets better just to ease into it', 'Thanks, Do you think, i should include any exercise for front delts ? Cause i am confused Front delts gets a work in a lot of chest exercises.', 'Overhead shoulder press would be good. I think you are right that you already hit the front delts with chest but shoulder presses would be a good addition. I think best is rear delts which you already are doing', 'Yeah i can add shoulder press for sure. Thanks man.\nHow long have you been lifting ?', '15+ years')