import os
import praw
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Authenticating with Reddit
def connect_to_reddit():
    reddit = praw.Reddit(
        client_id=os.getenv('CLIENT_ID'),
        client_secret=os.getenv('CLIENT_SECRET'),
        username=os.getenv('USERNAME'),
        password=os.getenv('PASSWORD'),
        user_agent="reddit scraper",
    )
    return reddit