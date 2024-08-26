"""
File: constants.py

Author: Anjola Aina
Date Modified: August 25th, 2024

Description:
    This file contains all the necessary constant classes that are used in the creation of the model, including scraper constants from Reddit.
    All of the constant classes in this file are immutable, to avoid them from being changed "accidently" in other codes. Therefore, to change the
    constant attributes, they must be changed from this file.

Classes:
    REDDIT_SCRAPER_CONSTANTS: All constants related to scraping relevant information from reddit.
"""
from dataclasses import dataclass

# Class Constants
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
    SUBREDDITS: tuple[str] = ('loseit', '1200isplenty', 'caloriecounting', 'cico', 'intermittentfasting', 'nutrition')
    KEYWORDS_CALORIE_COUNTING: tuple[str] = ('calorie counting app', 'best calorie counting app', 'calorie count app', 'best app')    
    SEARCH_QUERIES: tuple[str] = (' OR '.join(KEYWORDS_CALORIE_COUNTING),)
    
# Regular Constants
KEYWORDS_APPS = ('mfp', 'myfitnesspal', 'lose it', 'loseit', 'loseit!', 'cronometer', 'macrofactor', 'fatsecret', 'fat secret', 'mynetdiary', 'my net diary', 'yazio', 'lifesum')