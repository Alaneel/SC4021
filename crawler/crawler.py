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

def combine_datasets(reddit_df, twitter_df=None, output_filename=None):
    """
    Process Reddit data into a unified dataset.

    Args:
        reddit_df (DataFrame): DataFrame containing Reddit data
        twitter_df (DataFrame, optional): Parameter kept for backward compatibility
        output_filename (str, optional): Path to save the combined CSV file

    Returns:
        DataFrame: Processed dataset
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

    # Create and save combined DataFrame
    combined_df = pd.DataFrame(combined_data)
    if output_filename:
        combined_df.to_csv(output_filename, index=False)
        print(f"Saved dataset with {len(combined_df)} records to {output_filename}")

    return combined_df