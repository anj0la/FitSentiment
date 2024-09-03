"""
File: reddit_scraper.py

Author: Anjola Aina
Date Modified: August 25th, 2024

This file contains all the necessary functions used to scrape relevant information from Reddit.
NOTE: You only need run the scraper once to get new data and use it to train the mode, and a corpus of extracted data is already available in data/corpus.csv.

Functions:
    scrape_comments(int) -> list[str]: Scrapes comments from specific subredits that are focused on workout splits.
    run_scraper() -> None: Runs the reddit scraper, saving its information into a csv file to be futher processed.
"""
from praw import models
from fitsentiment.connector import connect_to_reddit
from constants.constants import REDDIT_SCRAPER_CONSTANTS, KEYWORDS_APPS, KEYWORDS_FEATURES
import csv

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
        run_scraper() -> None
    """
    
    def __init__(self, subreddits: list[str], search_queries: list[str], limit: int):
        self.reddit = connect_to_reddit()
        self.subreddits: list[str] = subreddits
        self.search_queries: list[str] = search_queries
        self.limit: int = limit
        
    def scrape_comments(self, sort: str ='relevance') -> list[str]:
        """
        Scrapes comments from the specified subreddits.

        Args:
            sort (str, optional): Determines how the submissions are grabbed. Can be sorted by "relevance", "hot", "top", "new", or "comments". Defaults to relevance.

            Returns:
                list: A list of scraped comments from the specified subreddits.
        """
        all_comments = [] # Stores all comments scraped from specificed subreddits
        for subreddit_name in self.subreddits:
            subreddit_instance = self.reddit.subreddit(subreddit_name)
            # Go through all the search queries to scrape comments
            for search_query in self.search_queries: 
                for submission in subreddit_instance.search(query=search_query, sort=sort, limit=self.limit):
                    comments = self._get_comments(submission)
                    # Extend the corpus to keep the dimension of the list (1D list, where each element is a string)
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
            if any(word in comment.body for word in KEYWORDS_APPS) and any(word in comment.body for word in KEYWORDS_FEATURES):
                comments.append(comment.body)
        return comments
    
    def run_scraper(self, path: str) -> None:
        """
        Runs the Reddit scraper.

        Args:
            path (str): The path name of the file to save the scraped information to.
        """
        corpus = self.scrape_comments()
        print(len(corpus))
        fields = ['text']
        rows = [{'text': sentence} for sentence in corpus]
        with open(path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            csv_file.close()  

# Running the scraper 
file_path = 'data/corpus.csv'
reddit_scraper = RedditScraper(subreddits=REDDIT_SCRAPER_CONSTANTS.SUBREDDITS, search_queries=REDDIT_SCRAPER_CONSTANTS.SEARCH_QUERIES, limit=999)
reddit_scraper.run_scraper(path=file_path)
