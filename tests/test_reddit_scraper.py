"""
File: test_reddit_scraper.py

Author: Anjola Aina
Date Modified: May 29th, 2024

Description:
    This file contains all the necessary functions to test the functionality of the reddit scraper.
    Currently, the test file has a coverage of 100%.
"""

from unittest import mock
from unittest.mock import MagicMock
from fitsentiment.reddit_scraper import RedditScraper
from constants.test_constants import REDDIT_SCRAPER_RESPONSES
from praw import models

@mock.patch('fitsentiment.reddit_scraper.connect_to_reddit')
def test_scrape_comments(connect_to_reddit_fn):
    # creating a mock reddit, subreddit and submission instances
    reddit_fn = MagicMock()
    connect_to_reddit_fn.return_value = reddit_fn
    
    subreddit_fn = MagicMock()
    reddit_fn.subreddit.return_value = subreddit_fn

    submission_fn = MagicMock()
    submission_fn.comments.list.return_value = [
            MagicMock(body=REDDIT_SCRAPER_RESPONSES.TEST_CORPUS[0]),
            MagicMock(body=REDDIT_SCRAPER_RESPONSES.TEST_CORPUS[1]),
        ]
    subreddit_fn.search.return_value = [submission_fn]
    
    # defining the constants for testing
    SUBREDDITS = ['testsubreddit']
    SEARCH_QUERIES = ['testquery']
    
    # Instantiating the RedditScraper class
    reddit_scraper = RedditScraper(subreddits=SUBREDDITS, search_queries=SEARCH_QUERIES, limit=5)

    # calling the function
    comments = reddit_scraper.scrape_comments()

    # assertions
    reddit_fn.subreddit.assert_called_once_with('testsubreddit')
    subreddit_fn.search.assert_called_once_with(query='testquery', sort='hot', limit=5)
    assert len(comments) == REDDIT_SCRAPER_RESPONSES.LEN_CORPUS # equal to 2 (length of the comments)
    assert comments[0] == REDDIT_SCRAPER_RESPONSES.TEST_CORPUS[0]
    assert comments[1] == REDDIT_SCRAPER_RESPONSES.TEST_CORPUS[1]
        
def test__get_comments():
    # create a mock submission with a mock comment forest
    submission_fn = MagicMock(spec=models.Submission)
    comment_fn = MagicMock()
    comment_fn.body = REDDIT_SCRAPER_RESPONSES.TEST_COMMENT
    submission_fn.comments.list.return_value = [comment_fn]

    # instantiating the RedditScraper class
    reddit_scraper = RedditScraper(subreddits=[], search_queries=[], limit=0)

    # calling the private _get_comments method directly for testing
    comments = reddit_scraper._get_comments(submission_fn)

    # assertions
    assert len(comments) == REDDIT_SCRAPER_RESPONSES.LEN_CORPUS  # equal to 2 (length of the comments)
    assert comments[0] == REDDIT_SCRAPER_RESPONSES.TEST_SPLIT_COMMENT[0]
    assert comments[1] == REDDIT_SCRAPER_RESPONSES.TEST_SPLIT_COMMENT[1]
