from fitsentiment.reddit_scraper import RedditScraper # gonna save it to a csv file instead
from constants.constants import REDDIT_SCRAPER_CONSTANTS
from fitsentiment.preprocess_data import TextPipeline
from fitsentiment.model import LSTM

# Usage example 
reddit_scraper = RedditScraper(subreddits=REDDIT_SCRAPER_CONSTANTS.SUBREDDITS, search_queries=REDDIT_SCRAPER_CONSTANTS.SEARCH_QUERIES, limit=1)
corpus = reddit_scraper.scrape_comments()

text_pipeline = TextPipeline()
features, vocab = text_pipeline.fit(corpus)


