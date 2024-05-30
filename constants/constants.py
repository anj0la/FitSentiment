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
    SUBREDDITS: tuple[str] = ('workout', 'WorkoutRoutines', 'loseit', 'bodybuilding', 'xxfitness', 'naturalbodybuilding')
    KEYWORDS_SPLITS: tuple[str] = ('workout split', 'training split', 'weekly split')
    KEYWORDS_TYPES_SPLITS: tuple[str] = ('push pull legs', 'upper lower', 'full body', 'body part split', 'phat', 'bro split', 'arnold split', 'hybrid split')
    SEARCH_QUERIES: tuple[str] = (' OR '.join(KEYWORDS_SPLITS), ' OR '.join(KEYWORDS_TYPES_SPLITS))
    

# Regular Constants

WORKOUT_KEYWORDS: tuple[str] = ('leg', 'chest', 'back', 'arm', 'shoulder', 'biceps', 'triceps', 'glute', 'gluteus maximus', 'gluteus medius', 
                                    'quad', 'hamstring', 'calf', 'delts', 'front delts', 'side delts', 'rear delts', 'trap', 'lat', 'ab', 'abdominal', 
                                    'ab', 'adductor', 'abductor', 'forearm', 'oblique', 'core', 'pecs', 'pec major', 'pec minor', 'rhomboid', 'rotator cuff', 
                                    'spinal erectors', 'pectoral', 'trapizoids', 'teres major', 'teres minor', 'serratus anterior')
