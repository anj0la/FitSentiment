# test_reddit_scraper.py
import pytest
from dotenv import load_dotenv
import sys
print(sys.path)
sys.path.append('/Users/anjola/Desktop/FitSentiment/')
from src.reddit_scraper import scrape_comments, _get_comments
from utils.constants import KEYWORDS

# Load environment variables from .env file
load_dotenv()

# Define test cases for scrape_comments function
def test_scrape_comments_returns_list():
    # Call the function
    result = scrape_comments(limit=10)
    # Check if the result is a list
    assert isinstance(result, list)

# Define test cases for _get_comments function
def test_get_comments_returns_list():
    # Mock a submission object
    class MockSubmission:
        def __init__(self, comments):
            self.comments = comments
    
    # Mock comments
    comments = [
        type('Comment', (object,), {'body': 'This is a test comment'}),
        type('Comment', (object,), {'body': 'Another test comment'}),
    ]

    # Call the function
    result = _get_comments(MockSubmission(comments))
    # Check if the result is a list
    assert isinstance(result, list)

# Define test cases for keyword filtering and moderation values
def test_get_comments_filter_keywords_and_moderation():
    # Mock a submission object
    class MockSubmission:
        def __init__(self, comments):
            self.comments = comments
            self.comments.replace_more.return_value = None
    
    # Mock comments
    comments = [
        type('Comment', (object,), {'body': 'This is a test comment'}),
        type('Comment', (object,), {'body': 'Another test comment'}),
        type('Comment', (object,), {'body': 'I like running and lifting weights'}),
        type('Comment', (object,), {'body': '*** This is a moderation comment ***'}),
    ]

    # Call the function
    result = _get_comments(MockSubmission(comments))
    # Check if the result contains only comments with keywords and no moderation values
    assert all(keyword in comment.lower() for comment in result for keyword in KEYWORDS)
    assert all('***' not in comment.lower() for comment in result)
