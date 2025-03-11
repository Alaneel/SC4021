"""
X (Twitter) crawler for collecting Electric Vehicle opinions
"""
import tweepy
import pandas as pd
import time
import random
from datetime import datetime
import os
import logging
from tqdm import tqdm
import sys
import re

from .data_cleaner import clean_text
from config.app_config import (
    X_API_KEY,
    X_API_SECRET,
    X_BEARER_TOKEN,
    X_ACCESS_TOKEN,
    X_ACCESS_SECRET,
    X_USER_AGENT,
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


class XCrawler:
    """
    Crawler for collecting EV-related opinions from X (Twitter)
    """

    def __init__(self, api_key=None, api_secret=None, bearer_token=None,
                 access_token=None, access_secret=None, user_agent=None):
        """
        Initialize the X (Twitter) crawler

        Args:
            api_key (str): X API key
            api_secret (str): X API secret
            bearer_token (str): X Bearer Token
            access_token (str): X Access Token
            access_secret (str): X Access Token Secret
            user_agent (str): User agent for X API
        """
        self.api_key = api_key or X_API_KEY
        self.api_secret = api_secret or X_API_SECRET
        self.bearer_token = bearer_token or X_BEARER_TOKEN
        self.access_token = access_token or X_ACCESS_TOKEN
        self.access_secret = access_secret or X_ACCESS_SECRET
        self.user_agent = user_agent or X_USER_AGENT

        if not self.bearer_token:
            raise ValueError("X Bearer Token not provided. "
                             "Set X_BEARER_TOKEN environment variable "
                             "or pass it as a parameter.")

        # Initialize Tweepy client with bearer token for read-only access
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_secret,
            wait_on_rate_limit=True
        )

        self.data = []
        logger.info("X crawler initialized")

    def _process_tweet(self, tweet, search_query=None):
        """
        Process a tweet into a standardized format

        Args:
            tweet: Tweet object from Tweepy
            search_query (str): The search query that found this tweet

        Returns:
            dict: Processed tweet data
        """
        # Extract tweet data
        tweet_data = {
            'id': tweet.id,
            'type': 'tweet',
            'title': search_query or '',  # No title in tweets, use search query
            'text': tweet.text,
            'author': tweet.author_id,
            'created_utc': tweet.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'score': tweet.public_metrics.get('like_count', 0),
            'subreddit': 'x',  # Use 'x' instead of subreddit for platform indicator
            'url': f"https://twitter.com/user/status/{tweet.id}",
            'num_comments': tweet.public_metrics.get('reply_count', 0),
            'retweet_count': tweet.public_metrics.get('retweet_count', 0),
            'quote_count': tweet.public_metrics.get('quote_count', 0),
            'hashtags': ','.join(re.findall(r'#(\w+)', tweet.text))
        }

        if hasattr(tweet, 'entities') and tweet.entities is not None:
            # Extract mentions
            mentions = []
            if 'mentions' in tweet.entities:
                for mention in tweet.entities['mentions']:
                    mentions.append(mention['username'])

            tweet_data['mentions'] = ','.join(mentions)

            # Extract URLs
            urls = []
            if 'urls' in tweet.entities:
                for url in tweet.entities['urls']:
                    urls.append(url['expanded_url'])

            tweet_data['urls'] = ','.join(urls)

        return tweet_data

    def search_tweets(self, query, max_results=100, skip_existing=True):
        """
        Search X for tweets matching a query

        Args:
            query (str): Search query
            max_results (int): Maximum number of tweets to retrieve
            skip_existing (bool): Whether to skip tweets already retrieved

        Returns:
            int: Number of new items collected
        """
        logger.info(f"Searching X for: '{query}'...")
        initial_count = len(self.data)

        try:
            # Check if we have existing IDs to avoid duplicates
            existing_ids = set()
            if skip_existing and self.data:
                existing_ids = set(item['id'] for item in self.data)

            # Define tweet fields to retrieve
            tweet_fields = [
                'created_at', 'author_id', 'public_metrics',
                'entities', 'context_annotations'
            ]

            # Define expansion fields
            expansions = ['author_id']

            # Set up pagination
            collected = 0
            pagination_token = None

            # Twitter search query pagination loop
            with tqdm(total=max_results, desc=f"Collecting tweets for '{query}'") as pbar:
                while collected < max_results:
                    # Determine how many results to request in this batch (max 100 per request)
                    batch_size = min(100, max_results - collected)

                    try:
                        # Execute search with pagination
                        response = self.client.search_recent_tweets(
                            query=query,
                            max_results=batch_size,
                            tweet_fields=tweet_fields,
                            expansions=expansions,
                            next_token=pagination_token
                        )

                        # Check if we got results
                        if not response.data:
                            logger.info(f"No more results for query: '{query}'")
                            break

                        # Process tweets
                        for tweet in response.data:
                            # Skip if already processed
                            if tweet.id in existing_ids:
                                continue

                            # Process tweet
                            tweet_data = self._process_tweet(tweet, query)

                            # Add to collection
                            self.data.append(tweet_data)
                            collected += 1
                            pbar.update(1)

                            # Check if we've reached the limit
                            if collected >= max_results:
                                break

                        # Update pagination token for next request
                        if 'next_token' in response.meta:
                            pagination_token = response.meta['next_token']
                        else:
                            # No more results
                            break

                        # Add delay to respect rate limits
                        time.sleep(2)

                    except tweepy.TooManyRequests:
                        logger.warning("Rate limit exceeded. Waiting 15 minutes before retrying.")
                        time.sleep(900 + random.uniform(1, 60))  # Wait 15-16 minutes
                    except tweepy.TwitterServerError:
                        logger.warning("Twitter server error. Waiting 30 seconds before retrying.")
                        time.sleep(30 + random.uniform(1, 30))
                    except Exception as e:
                        logger.error(f"Error searching tweets: {str(e)}")
                        time.sleep(10)
                        break

            new_items = len(self.data) - initial_count
            logger.info(f"Collected {new_items} new tweets for '{query}'")
            return new_items

        except Exception as e:
            logger.error(f"Error in search_tweets: {str(e)}")
            return 0

    def search_multiple_queries(self, queries, max_results_per_query=100):
        """
        Search X for multiple queries

        Args:
            queries (list): List of search queries
            max_results_per_query (int): Maximum tweets per query

        Returns:
            int: Total number of items collected
        """
        initial_count = len(self.data)

        # Track progress with tqdm
        total_queries = len(queries)
        with tqdm(total=total_queries, desc="Searching X") as pbar:
            for query in queries:
                self.search_tweets(query, max_results=max_results_per_query)
                pbar.update(1)

                # Save intermediate results every 5 iterations
                if len(self.data) % 500 == 0 and len(self.data) > 0:
                    self.save_intermediate_results()

                # Add a delay between queries
                delay = 10 + random.uniform(2, 8)  # 10-18 seconds between queries
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
        filename = os.path.join(RAW_DATA_DIR, f"x_ev_opinions_intermediate_{timestamp}.csv")

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
            filename = os.path.join(RAW_DATA_DIR, f"x_ev_opinions_{datetime.now().strftime('%Y%m%d')}.csv")

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
        logger.info(f"Total records: {total_records}")

        # Word count stats
        df['word_count'] = df['text'].str.split().str.len()
        total_words = df['word_count'].sum()
        unique_words = len(set(' '.join(df['text'].dropna()).split()))

        logger.info(f"Total words: {total_words}")
        logger.info(f"Unique words: {unique_words}")
        logger.info(f"Average words per record: {total_words / total_records:.1f}")

        return df