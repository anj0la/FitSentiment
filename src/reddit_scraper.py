# Import statements
import praw
import os
from dotenv import load_dotenv
from constants import KEYWORDS, SUBREDDITS

# Load environment variables from .env file
load_dotenv()

# Authenticating with Reddit
reddit = praw.Reddit(
    client_id=os.getenv('CLIENT_ID'),
    client_secret=os.getenv('CLIENT_SECRET'),
    username=os.getenv('USERNAME'),
    password=os.getenv('PASSWORD'),
    user_agent="reddit scraper",
)

# Functions to scrape posts and comments
def scrape_comments(limit):
    all_comments = [] # creates the training corpus
    for subreddit in SUBREDDITS:
        subreddit_instance = reddit.subreddit(subreddit)
        for submission in subreddit_instance.hot(limit=limit):
            if any(keyword in submission.title.lower() or keyword in submission.selftext.lower() for keyword in KEYWORDS):
                comments = _get_comments(submission)
                # we extend the comment instead of appending it to keep the dimension of the list to one
                all_comments.extend(comments)
    return all_comments

def _get_comments(submission):
    comments = []
    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        if any(keyword in comment.body.lower() for keyword in KEYWORDS):
            # removing all moderation cmments
            if '***' not in comment.body.lower():
                # splitting replies into their own separate sentences
                comments.extend(comment.body.split('\n\n'))
    return comments

# Usage example
corpus = scrape_comments(limit=50)
# print('corpus: ', corpus)
print('corpus length: ', len(corpus))

            
