# Import statements
import string
from praw.models import Submission
from connector import connect_to_reddit
from utils.constants import ALL_KEYWORDS, OPINION_INDICATORS, PHRASES, SUBREDDITS, SEARCH_QUERIES

def scrape_comments(limit: int) -> list:
    """
    Scrapes comments from specific subreddits focused on workouts.

    Args:
        limit (int): the number of submissions (maximum = 1000)

    Returns:
        list: A list of scraped comments from specific subreddits focused on workouts.
    """
    all_comments = [] # creates the training corpus
    reddit = connect_to_reddit() # connect to the reddit api
    for subreddit_name in SUBREDDITS:
        subreddit_instance = reddit.subreddit(subreddit_name)
        # going through workout splits (i.e., weekly / training split) and specific splits (i.e., full body) 
        for search_query in SEARCH_QUERIES:            
            for submission in subreddit_instance.search(query=search_query, sort='hot', limit=limit):
                comments = _get_comments(submission)
                # we extend the comment instead of appending it to keep the dimension of the list to one
                all_comments.extend(comments)
    return all_comments

def _get_comments(submission: Submission) -> list:
    """
    Retrieves comments from a specific submission from a specific subreddit focused on workouts.

    Args:
        submission (Submission): a submission (i.e., a reddit post) from a specific subreddit.

    Returns:
        list: a list of comments from the specific submission.
    """
    comments = []
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        comments.extend(comment.body.split('\n\n'))
    return comments

# Usage example
corpus = scrape_comments(limit=10)
# print('corpus: ', corpus)
print('corpus length: ', len(corpus))
print('print a subset of the corpus \n\n', corpus[0:3])