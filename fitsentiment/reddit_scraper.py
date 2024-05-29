"""
File: reddit_scraper.py

Author: Anjola Aina
Date Modified: May 29th, 2024

Description:
    This file contains all the necessary functions used to scrape relevant information from Reddit.
    There is one public function, scrape_comments, which uses a private function to grab comments from a specific submission (i.e., post) from a subreddit.

Functions:
    scrape_comments(int) -> list: Scrapes comments from specific subredits that are focused on workout splits.

Sources: N/A
"""
from praw import models
from connector import connect_to_reddit
from constants.constants import REDDIT_SCRAPER_CONSTANTS

def scrape_comments(limit: int) -> list:
    """
    Scrapes comments from specific subreddits focused on workout splits.

    Args:
        limit (int): The number of submissions (maximum limit = 1000).

    Returns:
        list: A list of scraped comments from specific subreddits focused on workouts.
    """
    all_comments = [] # creates the training corpus
    reddit = connect_to_reddit() # connect to the reddit api
    for subreddit_name in REDDIT_SCRAPER_CONSTANTS.SUBREDDITS:
        subreddit_instance = reddit.subreddit(subreddit_name)
        # going through workout splits (i.e., weekly / training split) and specific splits (i.e., full body) 
        for search_query in REDDIT_SCRAPER_CONSTANTS.SEARCH_QUERIES:            
            for submission in subreddit_instance.search(query=search_query, sort='hot', limit=limit):
                comments = _get_comments(submission)
                # we extend the comment instead of appending it to keep the dimension of the list to one
                all_comments.extend(comments)
    return all_comments

def _get_comments(submission: models.Submission) -> list:
    """
    Retrieves comments from a specific submission from a specific subreddit focused on workouts.

    Args:
        submission (Submission): A submission (i.e., a reddit post) from a specific subreddit.

    Returns:
        list: A list of comments from the specific submission.
    """
    comments = []
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        comments.extend(comment.body.split('\n\n'))
    return comments

# Usage example (TODO: delete later)
corpus = scrape_comments(limit=5)
# print('corpus: ', corpus)
print('corpus length: ', len(corpus))
print('print a subset of the corpus \n\n', corpus[180:200])