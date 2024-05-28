# Imports
from dotenv import load_dotenv
from src.connector import connect_to_reddit
from src.reddit_scraper import scrape_comments, _get_comments
from utils.constants import KEYWORDS