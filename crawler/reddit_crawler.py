"""
Reddit crawler for collecting Electric Vehicle opinions
"""
import praw
import pandas as pd
import time
import random
from datetime import datetime
import os
import logging
from tqdm import tqdm
import sys
from prawcore.exceptions import ResponseException, TooManyRequests, RequestException

from .data_cleaner import clean_text
from config.app_config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    RAW_DATA_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class RedditCrawler:
    """
    Crawler for collecting EV-related opinions from Reddit
    """

    def __init__(self, client_id=None, client_secret=None, user_agent=None):
        """
        Initialize the Reddit crawler

        Args:
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret
            user_agent (str): User agent for Reddit API
        """
        self.client_id = client_id or REDDIT_CLIENT_ID
        self.client_secret = client_secret or REDDIT_CLIENT_SECRET
        self.user_agent = user_agent or REDDIT_USER_AGENT

        if not self.client_id or not self.client_secret:
            raise ValueError("Reddit API credentials not provided. "
                             "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables "
                             "or pass them as parameters.")

        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )

        self.data = []
        logger.info("Reddit crawler initialized")

    def crawl_subreddit(self, subreddit_name, search_query, limit=100, skip_existing=True):
        """
        Crawl a subreddit for posts and comments matching a query

        Args:
            subreddit_name (str): Name of the subreddit to crawl
            search_query (str): Search query to use
            limit (int): Maximum number of submissions to retrieve
            skip_existing (bool): Whether to skip posts already retrieved

        Returns:
            int: Number of new items collected
        """
        logger.info(f"Crawling r/{subreddit_name} for '{search_query}'...")
        initial_count = len(self.data)

        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Check if subreddit exists and is accessible
            subreddit.fullname

            # Get existing post IDs to avoid duplicates
            existing_ids = set()
            if skip_existing and self.data:
                existing_ids = set(item['id'] for item in self.data)

            # Search for submissions with error handling and backoff
            try:
                search_results = list(subreddit.search(search_query, limit=limit))
            except (ResponseException, RequestException) as e:
                logger.warning(f"Error during search, implementing backoff: {str(e)}")
                time.sleep(30 + random.uniform(0, 10))  # Add significant delay
                try:
                    search_results = list(subreddit.search(search_query, limit=limit))
                except Exception as e:
                    logger.error(f"Second attempt failed: {str(e)}")
                    return 0

            for i, submission in enumerate(search_results):
                try:
                    # Skip if already processed
                    if submission.id in existing_ids:
                        continue

                    # Extract submission data
                    post_data = {
                        'id': submission.id,
                        'type': 'post',
                        'title': submission.title,
                        'text': submission.selftext,
                        'author': str(submission.author),
                        'created_utc': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'score': submission.score,
                        'subreddit': subreddit_name,
                        'url': submission.url,
                        'num_comments': submission.num_comments
                    }

                    # Only add if there's actual text content
                    if len(post_data['text'].strip()) > 0 or len(post_data['title'].strip()) > 0:
                        self.data.append(post_data)

                    # Get comments (limit to top-level to avoid rate limiting)
                    if i % 3 == 0:  # Only get comments for every third post to reduce API calls
                        try:
                            submission.comments.replace_more(limit=0)
                            for comment in submission.comments.list()[:20]:  # Limit to first 20 comments
                                if hasattr(comment, 'body') and comment.id not in existing_ids:
                                    comment_data = {
                                        'id': comment.id,
                                        'type': 'comment',
                                        'title': '',  # Comments don't have titles
                                        'text': comment.body,
                                        'author': str(comment.author),
                                        'created_utc': datetime.fromtimestamp(comment.created_utc).strftime(
                                            '%Y-%m-%d %H:%M:%S'),
                                        'score': comment.score,
                                        'subreddit': subreddit_name,
                                        'url': f"https://www.reddit.com{submission.permalink}{comment.id}/",
                                        'parent_id': comment.parent_id
                                    }

                                    # Only add if there's actual text content
                                    if len(comment_data['text'].strip()) > 20:  # At least 20 chars
                                        self.data.append(comment_data)
                        except Exception as e:
                            logger.warning(f"Error fetching comments for submission {submission.id}: {str(e)}")

                    # Sleep to respect rate limits - significant delay
                    delay = 5 + random.uniform(2, 5)  # 5-10 seconds between submissions
                    time.sleep(delay)

                except Exception as e:
                    logger.warning(f"Error processing submission: {str(e)}")
                    time.sleep(5)  # Add delay on errors

            new_items = len(self.data) - initial_count
            logger.info(f"Collected {new_items} new items from r/{subreddit_name} for '{search_query}'")
            return new_items

        except TooManyRequests as e:
            logger.error(f"Rate limited by Reddit API. Waiting 60 seconds before retrying.")
            time.sleep(60 + random.uniform(0, 10))  # Significant delay for rate limiting
            return 0
        except Exception as e:
            logger.error(f"Error crawling r/{subreddit_name}: {str(e)}")
            time.sleep(10)  # Add delay on errors
            return 0

    def crawl_multiple(self, subreddits, queries, limit_per_query=50):
        """
        Crawl multiple subreddits with multiple queries

        Args:
            subreddits (list): List of subreddit names
            queries (list): List of search queries
            limit_per_query (int): Maximum posts per query

        Returns:
            int: Total number of items collected
        """
        initial_count = len(self.data)

        # Track progress with tqdm
        total_combinations = len(subreddits) * len(queries)
        with tqdm(total=total_combinations, desc="Crawling Reddit") as pbar:
            for subreddit in subreddits:
                for query in queries:
                    self.crawl_subreddit(subreddit, query, limit=limit_per_query)
                    pbar.update(1)

                    # Save intermediate results every 5 iterations
                    if len(self.data) % 500 == 0 and len(self.data) > 0:
                        self.save_intermediate_results()

                    # Add a significant delay between queries
                    delay = 30 + random.uniform(5, 15)  # 30-45 seconds between queries
                    time.sleep(delay)

        new_items = len(self.data) - initial_count
        logger.info(f"Total new items collected: {new_items}")
        return new_items

    def save_intermediate_results(self):
        """Save intermediate results to avoid data loss"""
        if not self.data:
            return

        # Create timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RAW_DATA_DIR, f"reddit_ev_opinions_intermediate_{timestamp}.csv")

        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} records to {filename}")

    def save_to_csv(self, filename=None):
        """
        Save the collected data to a CSV file

        Args:
            filename (str): Output filename

        Returns:
            pandas.DataFrame: The saved dataframe
        """
        if not self.data:
            logger.warning("No data to save")
            return None

        if filename is None:
            filename = os.path.join(RAW_DATA_DIR, f"reddit_ev_opinions_{datetime.now().strftime('%Y%m%d')}.csv")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        df = pd.DataFrame(self.data)

        # Remove duplicates
        df = df.drop_duplicates(subset=['id'])

        # Save to CSV
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} records to {filename}")

        # Print some statistics
        total_records = len(df)
        posts = df[df['type'] == 'post']
        comments = df[df['type'] == 'comment']

        logger.info(f"Total records: {total_records}")
        logger.info(f"Posts: {len(posts)}")
        logger.info(f"Comments: {len(comments)}")
        logger.info(f"Unique subreddits: {df['subreddit'].nunique()}")

        # Word count stats
        df['word_count'] = df['text'].str.split().str.len()
        total_words = df['word_count'].sum()
        unique_words = len(set(' '.join(df['text'].dropna()).split()))

        logger.info(f"Total words: {total_words}")
        logger.info(f"Unique words: {unique_words}")
        logger.info(f"Average words per record: {total_words / total_records:.1f}")

        return df