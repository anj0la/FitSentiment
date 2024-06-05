"""
File: reddit_scraper.py

Author: Anjola Aina
Date Modified: May 30th, 2024

Description:
    This file contains all the necessary functions used to scrape relevant information from Reddit.
    NOTE: You only need run the scraper once to get new data and use it to train the mode, and a corpus of extracted data is already available in data/corpus.csv.

Functions:
    scrape_comments(int) -> list: Scrapes comments from specific subredits that are focused on workout splits.
    run_scraper() -> None: Runs the reddit scraper, saving its information into a csv file to be futher processed.
"""
from praw import models
from fitsentiment.connector import connect_to_reddit
from constants.constants import REDDIT_SCRAPER_CONSTANTS
from utils.label_data import create_text_label_rows
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
            comments.append(comment.body)
        return comments
    
    def run_scraper(self, file_path):
        """
        Runs the Reddit scraper.

        Args:
            file_path (str): The pathname of the file to save the scraped information to.
        """
        corpus = self.scrape_comments()
        print(len(corpus))
        fields = ['text', 'label']
        rows = create_text_label_rows(corpus=corpus)
        with open(file_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            csv_file.close()  

# Running the scraper 
file_path = 'data/corpus.csv'
reddit_scraper = RedditScraper(subreddits=REDDIT_SCRAPER_CONSTANTS.SUBREDDITS, search_queries=REDDIT_SCRAPER_CONSTANTS.SEARCH_QUERIES, limit=10)
reddit_scraper.run_scraper()