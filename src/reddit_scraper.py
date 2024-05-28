# Import statements
from connector import connect_to_reddit
from utils.constants import KEYWORDS, SUBREDDITS

# Functions to scrape posts and comments
def scrape_comments(limit):
    all_comments = [] # creates the training corpus
    reddit = connect_to_reddit() # connect to the reddit api
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
corpus = scrape_comments(limit=1)
# print('corpus: ', corpus)
print('corpus length: ', len(corpus))

            
