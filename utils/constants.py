# MOVE INTO CONSTANTS FOR TEST
TEST_CORPUS =   ['I want to use this so bad but I feel it should have a rest day Thursday then do the rest, do you think that will help you more?', 'You absolutely must incorporate squats into your leg workout as well as deadlifts (either also on leg day or on back day). Those are two of the three most important and effective lifts that hit well beyond your legs', 'Taking rest on friday, after one complete cycle.', "I can't do squats and deadlifts due to lower back injury but i will add them once i feel better. \nThanks man", 'But what if your other muscles are actually sore due to progressive overload? Assuming that’s what you’re doing and you are trying hard to build up that muscle!', 'Apologies - makes total sense then. Be safe. Maybe try hack squats before you even do start squats when your back gets better just to ease into it', 'Thanks, Do you think, i should include any exercise for front delts ? Cause i am confused Front delts gets a work in a lot of chest exercises.', 'Overhead shoulder press would be good. I think you are right that you already hit the front delts with chest but shoulder presses would be a good addition. I think best is rear delts which you already are doing', 'Yeah i can add shoulder press for sure. Thanks man.\nHow long have you been lifting ?', '15+ years']

# Subreddits to search for keywords
SUBREDDITS = ['workout', 'WorkoutRoutines', 'loseit', 'bodybuilding', 'xxfitness', 'naturalbodybuilding']

# Keywords to filter the comments to things related to workout splits
KEYWORDS_SPLITS = ['workout split', 'training split', 'weekly split']
KEYWORDS_TYPES_SPLITS = ['push pull legs', 'upper lower', 'full body', 'body part split', 'phat', 'bro split', 'arnold split', 'hybrid split']
ALL_KEYWORDS = KEYWORDS_SPLITS + KEYWORDS_TYPES_SPLITS

# Search queries to use to search the above subreddits
SEARCH_QUERIES = [' OR '.join(KEYWORDS_SPLITS), ' OR '.join(KEYWORDS_TYPES_SPLITS)]

# Phrases to further filter out comments to get personal opinions
PHRASES = [
    'i prefer', 'i like', 'i love', 'i hate', 'i dislike', 'my favorite', 
    'ive been doing', 'im loving', 'im hating', 'feels high paced', 
    'working for me', 'not working for me', 'i think', 'in my opinion', 
    'ive found', 'i always', 'i never', 'my experience', 'my goto', 'i chose', 
    'i decided', 'i started', 'ive tried', 'i recommend', 'i suggest'
]

# Personal pronouns and opinion indicators
OPINION_INDICATORS = [
    'i', 'my', 'me', 'mine', 'we', 'our', 'us', 'ours'
]