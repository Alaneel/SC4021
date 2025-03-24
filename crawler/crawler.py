# crawlers.py
"""Social media crawler classes for Reddit and Twitter."""

import praw
import tweepy
import pandas as pd
import time
import json
from datetime import datetime
from tqdm import tqdm
from config import STREAMING_PLATFORMS


class RedditCrawler:
    """Class for crawling Reddit posts and comments related to streaming services."""

    def __init__(self, client_id, client_secret, user_agent):
        """Initialize the Reddit crawler with API credentials."""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.data = []

    def crawl_subreddit(self, subreddit_name, limit=1000, search_query=None):
        """
        Crawl posts and comments from a subreddit.

        Args:
            subreddit_name (str): Name of the subreddit to crawl
            limit (int): Maximum number of posts to retrieve
            search_query (str, optional): Search query to filter posts
        """
        subreddit = self.reddit.subreddit(subreddit_name)

        if search_query:
            submissions = subreddit.search(search_query, limit=limit)
        else:
            submissions = subreddit.top(time_filter="all", limit=limit)

        for submission in tqdm(submissions, desc=f"Crawling r/{subreddit_name}"):
            # Extract post data
            post_data = {
                'id': submission.id,
                'title': submission.title,
                'text': submission.selftext,
                'score': submission.score,
                'created_utc': datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'author': str(submission.author),
                'num_comments': submission.num_comments,
                'subreddit': subreddit_name,
                'permalink': submission.permalink,
                'type': 'submission',
                'platform': self._detect_streaming_platform(submission.title + " " + submission.selftext)
            }

            if post_data['text'] and len(post_data['text']) > 20:  # Filter out empty or very short posts
                self.data.append(post_data)

            # Get comments
            submission.comments.replace_more(limit=5)  # Limit comment depth for efficiency
            for comment in submission.comments.list():
                if not comment.body or len(comment.body) < 20:  # Skip short comments
                    continue

                comment_data = {
                    'id': comment.id,
                    'text': comment.body,
                    'score': comment.score,
                    'created_utc': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    'author': str(comment.author),
                    'parent_id': comment.parent_id,
                    'subreddit': subreddit_name,
                    'permalink': submission.permalink + comment.id + '/',
                    'type': 'comment',
                    'platform': self._detect_streaming_platform(comment.body)
                }
                self.data.append(comment_data)

    def _detect_streaming_platform(self, text):
        """
        Detect which streaming platform is being discussed in the text.

        Args:
            text (str): Text to analyze

        Returns:
            str: Detected platform name or 'general' if none found
        """
        text = text.lower()
        for platform, keywords in STREAMING_PLATFORMS.items():
            for keyword in keywords:
                if keyword in text:
                    return platform
        return 'general'  # If no specific platform is mentioned

    def save_to_csv(self, filename):
        """
        Save crawled data to CSV file.

        Args:
            filename (str): Path to save the CSV file

        Returns:
            DataFrame: Pandas DataFrame of the saved data
        """
        df = pd.DataFrame(self.data)
        df.drop_duplicates(subset=['id'], inplace=True)
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} records to {filename}")
        return df

    def save_to_json(self, filename):
        """
        Save crawled data to JSON file.

        Args:
            filename (str): Path to save the JSON file
        """
        with open(filename, 'w') as f:
            json.dump(self.data, f)
        print(f"Saved {len(self.data)} records to {filename}")


class TwitterCrawler:
    """Class for crawling Twitter posts related to streaming services."""

    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret, bearer_token):
        """Initialize the Twitter crawler with API credentials."""
        self.client = tweepy.Client(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            bearer_token=bearer_token
        )
        self.data = []

    def crawl_tweets(self, query, max_results=100, limit=5000):
        """
        Crawl tweets matching the query.

        Args:
            query (str): Search query for tweets
            max_results (int): Maximum results per API call
            limit (int): Total maximum number of tweets to retrieve
        """
        total_crawled = 0
        next_token = None

        with tqdm(total=limit, desc=f"Crawling tweets for '{query}'") as pbar:
            while total_crawled < limit:
                try:
                    # API call with pagination
                    response = self.client.search_recent_tweets(
                        query=query,
                        max_results=min(max_results, limit - total_crawled),
                        next_token=next_token,
                        tweet_fields=['created_at', 'public_metrics', 'author_id', 'lang']
                    )

                    if not response.data:
                        print(f"No more tweets found for query: {query}")
                        break

                    # Process tweet data
                    for tweet in response.data:
                        if tweet.lang != 'en':  # Filter non-English tweets
                            continue

                        tweet_data = {
                            'id': tweet.id,
                            'text': tweet.text,
                            'created_at': tweet.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                            'author_id': tweet.author_id,
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'reply_count': tweet.public_metrics['reply_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'query': query,
                            'platform': self._detect_streaming_platform(tweet.text)
                        }
                        self.data.append(tweet_data)

                    total_crawled += len(response.data)
                    pbar.update(len(response.data))

                    # Get next pagination token
                    if 'next_token' in response.meta:
                        next_token = response.meta['next_token']
                    else:
                        print(f"No more pagination tokens for query: {query}")
                        break

                    # Rate limit handling
                    time.sleep(3)  # Respect Twitter API rate limits

                except Exception as e:
                    print(f"Error crawling tweets: {str(e)}")
                    time.sleep(60)  # Wait longer if there's an error

        print(f"Crawled {len(self.data)} tweets for query: {query}")

    def _detect_streaming_platform(self, text):
        """
        Detect which streaming platform is being discussed in the text.

        Args:
            text (str): Text to analyze

        Returns:
            str: Detected platform name or 'general' if none found
        """
        text = text.lower()
        for platform, keywords in STREAMING_PLATFORMS.items():
            for keyword in keywords:
                if keyword in text:
                    return platform
        return 'general'  # If no specific platform is mentioned

    def save_to_csv(self, filename):
        """
        Save crawled data to CSV file.

        Args:
            filename (str): Path to save the CSV file

        Returns:
            DataFrame: Pandas DataFrame of the saved data
        """
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} records to {filename}")
        return df

    def save_to_json(self, filename):
        """
        Save crawled data to JSON file.

        Args:
            filename (str): Path to save the JSON file
        """
        with open(filename, 'w') as f:
            json.dump(self.data, f)
        print(f"Saved {len(self.data)} records to {filename}")


def combine_datasets(reddit_df, twitter_df=None, output_filename=None):
    """
    Combine Reddit and optional Twitter data into a single dataset.

    Args:
        reddit_df (DataFrame): DataFrame containing Reddit data
        twitter_df (DataFrame, optional): DataFrame containing Twitter data
        output_filename (str, optional): Path to save the combined CSV file

    Returns:
        DataFrame: Combined dataset
    """
    combined_data = []

    # Process Reddit data
    for _, row in reddit_df.iterrows():
        combined_data.append({
            'id': f"reddit_{row['id']}",
            'text': row['text'] if pd.notna(row['text']) else "",
            'title': row['title'] if 'title' in row and pd.notna(row['title']) else "",
            'created_at': row['created_utc'],
            'platform': row['platform'],
            'source': 'reddit',
            'score': row['score']
        })

    # Process Twitter data if provided
    if twitter_df is not None:
        for _, row in twitter_df.iterrows():
            combined_data.append({
                'id': f"twitter_{row['id']}",
                'text': row['text'],
                'title': "",
                'created_at': row['created_at'],
                'platform': row['platform'],
                'source': 'twitter',
                'score': row['like_count']
            })

    # Create and save combined DataFrame
    combined_df = pd.DataFrame(combined_data)
    if output_filename:
        combined_df.to_csv(output_filename, index=False)
        print(f"Saved combined dataset with {len(combined_df)} records to {output_filename}")

    return combined_df