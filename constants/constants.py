"""
File: constants.py

Author: Anjola Aina
Date Modified: May 29th, 2024

Description:
    This file contains all the necessary constant classes that are used in the creation of the model, including scraper constants from Reddit.
    All of the constant classes in this file are immutable, to avoid them from being changed "accidently" in other codes. Therefore, to change the
    constant attributes, they must be changed from this file.

Classes:
    REDDIT_SCRAPER_CONSTANTS: All constants related to scraping relevant information from reddit.
"""
from dataclasses import dataclass

# Class constants

@dataclass(frozen=True)
class REDDIT_SCRAPER_CONSTANTS:
    """
    This data class contains all of the constants related to the Reddit scraper. The class and its attributes are immutable,
    so that they cannot be changed after creation.
    
    Attributes:
        SUBREDDITS (tuple[str]): The subreddits to search for keywords.
        KEYWORDS_SPLITS (tuple[str]): The keywords to filter the search to Reddit posts related to workout splits.
        KEYWORDS_TYPES_SPLITS (tuple[str]): The keywords to filter the search to Reddit posts related to specific types of workout splits.
        SEARCH_QUERIES (tuple[str]): The search queries to use to search the subreddits for posts and comments.
    """
    SUBREDDITS: tuple[str] = ('workout', 'WorkoutRoutines', 'gainit', 'bodybuilding', 'xxfitness', 'naturalbodybuilding')
    KEYWORDS_SPLITS: tuple[str] = ('workout split', 'training split', 'weekly split')
    KEYWORDS_TRAINING_PREFERENCES: tuple[str] = ('favorite body part', 'favourite muscle group', 'favourite to train', 'most trained muscle group', 'training preferences')    
    KEYWORDS_GENERAL_WORKOUT = (
    'workout', 'workout routine', 'strength training', 'hypertrophy', 'powerlifting', 'bodybuilding', 'crossfit', 
    'HIIT', 'calisthenics', 'fitness journey','gym routine', 'exercise plan', 'muscle building', 'fat loss', 'endurance training',
    'personal records', 'progressive overload', 'fitness goals', 'workout plan',
    'gym equipment', 'training tips')
    SEARCH_QUERIES: tuple[str] = (' OR '.join(KEYWORDS_SPLITS), ' OR '.join(KEYWORDS_TRAINING_PREFERENCES))
    
# Regular Constants

# make class called Label Constants

LOWER_BODY_PARTS = ('legs', 'leg', 'quads', 'quad', 'hamstrings', 'hamstring', 'glutes', 'glute', 'calves', 'calf', 'adductors', 'adductor', 'abductors', 'abductor')
UPPER_BODY_PARTS = ('arms', 'arm', 'shoulders', 'shoulder', 'biceps', 'bicep', 'triceps', 'tricep', 'chest', 'forearms', 'forearm', 'delts', 'delt', 'back', 'lats', 'lat', 'traps', 'trap')
CORE_PARTS = ('abs', 'abdominals', 'obliques', 'core')
FULL_BODY_KEYWORDS = ('full', 'body' '3 days', '3 times', '3 day', '3 time')
UPPER_LOWER_KEYWORDS = ('4 day', '4 time', '4 times', '4 days', 'upper lower')
PUSH_PULL_LEGS_KEYWORDS = ('6 days', '6 times', '6 day', '6 time', 'push', 'pull', 'ppl', 'pplplr', 'pplppr')


WORKOUT_CLASSES = ('full body', 'upper lower', 'push pull legs', 'lower body', 'upper body', 'general fitness')
WORKOUT_CLASSES_VOCAB = {class_name: idx for idx, class_name in enumerate(WORKOUT_CLASSES)}
