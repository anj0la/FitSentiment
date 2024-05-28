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