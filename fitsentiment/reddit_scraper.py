"""
File: reddit_scraper.py

Author: Anjola Aina
Date Modified: May 30th, 2024

Description:
    This file contains all the necessary functions used to scrape relevant information from Reddit.
    There is one public function, scrape_comments, which uses a private function to grab comments from a specific submission (i.e., post) from a subreddit.

Functions:
    scrape_comments(int) -> list: Scrapes comments from specific subredits that are focused on workout splits.
"""
from praw import models
from fitsentiment.connector import connect_to_reddit
from constants.constants import REDDIT_SCRAPER_CONSTANTS

class RedditScraper:
    """
    This class is used to scrape revelant information from Reddit.
    
    Attributes:
        reddit (praw.Reddit): The reddit object which provides convenient access to Reddit's API.
        subreddits (list[str]): The list of subreddits to search from.
        search_queries (list[str]): The queries to search for each subreddit.
        limit (int): The limit of submissions (i.e, posts) per subreddit.
        
    Public Functions:
        scrape_comments(self, str): -> list
    """
    
    def __init__(self, subreddits: list[str], search_queries: list[str], limit: int):
        self.reddit = connect_to_reddit()
        self.subreddits: list[str] = subreddits
        self.search_queries: list[str] = search_queries
        self.limit: int = limit
        
    def scrape_comments(self, sort: str ='hot') -> list[str]:
        """
        Scrapes comments from the specified subreddits.

        Args:
            sort (str, optional): Determines how the submissions are grabbed. Can be sorted by "relevance", "hot", "top", "new", or "comments". Defaults to hot.

            Returns:
                list: A list of scraped comments from the specified subreddits.
        """
        all_comments = [] # creates the training corpus
        for subreddit_name in self.subreddits:
            subreddit_instance = self.reddit.subreddit(subreddit_name)
            # going through workout splits (i.e., weekly / training split) and specific splits (i.e., full body) 
            for search_query in self.search_queries:            
                for submission in subreddit_instance.search(query=search_query, sort=sort, limit=self.limit):
                    comments = self._get_comments(submission)
                    # we extend the comment instead of appending it to keep the dimension of the list to one
                    all_comments.extend(comments)
        return all_comments 
    
    def _get_comments(self, submission: models.Submission) -> list[str]:
        """
        Retrieves comments from a specific submission from a specific subreddit.

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
reddit_scraper = RedditScraper(subreddits=REDDIT_SCRAPER_CONSTANTS.SUBREDDITS, search_queries=REDDIT_SCRAPER_CONSTANTS.SEARCH_QUERIES, limit=1)
corpus = reddit_scraper.scrape_comments()
# print('corpus: ', corpus)
print('corpus length: ', len(corpus))
print('print a subset of the corpus \n\n', corpus[1:100])