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
    SUBREDDITS: tuple[str] = ('caloriecount', 'cico', 'loseit', 'intermittentfasting', '1200isplenty', '1500isplenty', 'nutrition')
    KEYWORDS_CALORIE_COUNTING: tuple[str] = ('calorie', 'calorie count', 'app', 'calorie counting app', 'best calorie counting app', 'calorie count app', 'best app')    
    SEARCH_QUERIES: tuple[str] = (' OR '.join(KEYWORDS_CALORIE_COUNTING),)
    
# Regular Constants
KEYWORDS_APPS = ('mfp', 'myfitnesspal', 'lose it', 'loseit', 'loseit!', 'cronometer', 'macrofactor', 'fatsecret', 'fat secret', 'mynetdiary', 'my net diary', 'yazio', 'lifesum')

KEYWORDS_FEATURES = (
    # Macronutrient Tracking
    'macronutrient tracking', 'macro tracking', 'macro counting', 'carb tracking', 'protein tracking', 'fat tracking', 'macro breakdown', 'macro', 'track macro'

    # Micronutrient Tracking
    'micronutrient tracking', 'micro tracking', 'vitamin tracking', 'mineral tracking', 'micronutrient', 'micro', 'track micro'

    # Calorie Tracking
    'calorie counting', 'calorie tracking', 'track calorie'
    
     # Food Diary / Diary Logging
    'food logging', 'food log', 'food entry', 'recipe entry', 'meal entry', 'add recipe', 'add food', 'recipe logging', 'meal logging', 'record food', 'record recipe', 'record meal', 'food diary',

    # Barcode Scanning
    'barcode scanning', 'barcode scanner', 'scanner', 'barcode', 'scanning'

    # Food / User Database
    'database', 'user submitted entries', 'entries', 'library', 'food list', 

    # Meal Planning
    'meal planning', 'meal plan', 'plan meals',

    # Diary Sharing
    'diary sharing',

    # Intake Limit
    'intake limit', 'calorie limit', 'daily limit', 'weekly limit',

    # Exercise Tracking
    'exercise tracking', 'workout tracking', 'exercise logging', 'workout logging', 'fitness tracking', 'track exercise', 'track workout',
    
    # Weight Tracking
    'weight tracking', 'weight log', 'weight entry', 'weight records', 'track weight'

    # Water Tracking
    'water tracking', 'track water'

    # Goals
    'goal setting', 'set goals', 

    # Custom Foods / Meals
    'custom food', 'create food', 'custom meal', 'create meal',

    # Custom Recipes
    'custom recipe', 'recipe builder', 'create recipe',

    # Custom Goals
    'custom goals', 'goal customization', 'create goals'

    # Custom Exercises / Workouts
    'custom exercises', 'custom workouts', 'workout customization', 'exercise customization',

    # Daily / Weekly Summary / Breakdown
    'daily summary', 'weekly summary', 'daily breakdown', 'weekly breakdown',

    # Activity Syncing
    'activity syncing', 'app sync', 'app integration',

    # Social Media Integration
    'social media', 'friends',

    # Reminders / Notifications
    'reminders',

    # Sleep Tracking
    'sleep tracking', 'track sleep',

    # Step Tracking
    'step tracking', 'step tracker', 'track step',

    # Calorie Burn Estimator
    'calorie burn',

    # Insights / Breakdown
    'health insights', 'nutrition insights', 'nutrition breakdown', 'nutrition strategies', 'stats', 'strategies', 'data insights',

    # Reports
    'reports',

    # AI Features
    'artificial intelligence', 'ai', 'AI',

    # Offline Availability
    'offline mode', 'offline',

    # Intermittent Fasting
    'intermittent fasting timer', 'fasting timer', 'fasting schedule', 'fasting tracker',

    # Gamification
    'weekly challenges', 'achievements', 'streaks', 'challenges', 'daily challenges', 'streak'
)
