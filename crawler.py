import praw
import tweepy
import pandas as pd
import re
import time
import json
from datetime import datetime
from tqdm import tqdm


class RedditCrawler:
    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.data = []

    def crawl_subreddit(self, subreddit_name, limit=1000, search_query=None):
        """Crawl posts and comments from a subreddit"""
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
        """Detect which streaming platform is being discussed"""
        platforms = {
            'netflix': ['netflix', 'netflix\'s'],
            'disney+': ['disney+', 'disney plus', 'disneyplus'],
            'hbo max': ['hbo max', 'hbomax', 'hbo'],
            'amazon prime': ['amazon prime', 'prime video', 'primevideo'],
            'hulu': ['hulu', 'hulu\'s'],
            'apple tv+': ['apple tv+', 'apple tv plus', 'appletv+'],
            'peacock': ['peacock'],
            'paramount+': ['paramount+', 'paramount plus']
        }

        text = text.lower()
        for platform, keywords in platforms.items():
            for keyword in keywords:
                if keyword in text:
                    return platform
        return 'general'  # If no specific platform is mentioned

    def save_to_csv(self, filename):
        """Save crawled data to CSV file"""
        df = pd.DataFrame(self.data)
        df.drop_duplicates(subset=['id'], inplace=True)
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} records to {filename}")
        return df

    def save_to_json(self, filename):
        """Save crawled data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.data, f)
        print(f"Saved {len(self.data)} records to {filename}")


class TwitterCrawler:
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret, bearer_token):
        self.client = tweepy.Client(consumer_key=consumer_key, consumer_secret=consumer_secret, access_token=access_token, access_token_secret=access_token_secret, bearer_token=bearer_token)
        self.data = []

    def crawl_tweets(self, query, max_results=100, limit=5000):
        """Crawl tweets matching the query"""
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
        """Detect which streaming platform is being discussed"""
        platforms = {
            'netflix': ['netflix', 'netflix\'s'],
            'disney+': ['disney+', 'disney plus', 'disneyplus'],
            'hbo max': ['hbo max', 'hbomax', 'hbo'],
            'amazon prime': ['amazon prime', 'prime video', 'primevideo'],
            'hulu': ['hulu', 'hulu\'s'],
            'apple tv+': ['apple tv+', 'apple tv plus', 'appletv+'],
            'peacock': ['peacock'],
            'paramount+': ['paramount+', 'paramount plus']
        }

        text = text.lower()
        for platform, keywords in platforms.items():
            for keyword in keywords:
                if keyword in text:
                    return platform
        return 'general'  # If no specific platform is mentioned

    def save_to_csv(self, filename):
        """Save crawled data to CSV file"""
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} records to {filename}")
        return df

    def save_to_json(self, filename):
        """Save crawled data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.data, f)
        print(f"Saved {len(self.data)} records to {filename}")


# Example usage
if __name__ == "__main__":
    # Reddit crawler
    reddit_crawler = RedditCrawler(
        client_id="2dKydEu9_1-1CJlfrCZh2g",
        client_secret="OYKcuCbwKItW9FYMDBDpj47sRCtvSQ",
        user_agent="Streaming Opinion Crawler 1.0 by /u/YOUR_USERNAME"
    )

    # Crawl multiple subreddits
    subreddits = ['Netflix', 'DisneyPlus', 'Hulu', 'PrimeVideo', 'cordcutters', 'StreamingBestOf']
    for subreddit in subreddits:
        reddit_crawler.crawl_subreddit(subreddit, limit=500)

    # Save Reddit data
    reddit_df = reddit_crawler.save_to_csv("reddit_streaming_data.csv")

    # Twitter crawler
    twitter_crawler = TwitterCrawler(consumer_key="VLn3KSpVzQ0shTdYpgpzW6uve", consumer_secret="pcNTtNz8sy2kcDnGDlpAxHTqZCSPyJmgjKpHYs7g0ugy88kewY", access_token="1555001250973749254-bGQDqO8HQVgDSGtcjuMi6hUKERLbZ1", access_token_secret="xnxXHvJZZ7K7i9a1oczTJK9EUwX9AxCReu5sFjKXNw256", bearer_token="AAAAAAAAAAAAAAAAAAAAAFd%2BzwEAAAAA%2FoftOrNppA7OcW%2BTA88j6PjoYWg%3DuNWHCCf1uIGjgkIOJuGYUXtby9ji8DTC7N2QoM49Ff3sUvc8jz")

    # Crawl tweets for different streaming services
    queries = [
        "Netflix review OR opinion OR thoughts -is:retweet",
        "Disney+ review OR opinion OR thoughts -is:retweet",
        "Hulu review OR opinion OR thoughts -is:retweet",
        "HBO Max review OR opinion OR thoughts -is:retweet",
        "Amazon Prime Video review OR opinion OR thoughts -is:retweet",
        "Apple TV+ review OR opinion OR thoughts -is:retweet"
    ]

    for query in queries:
        twitter_crawler.crawl_tweets(query, max_results=100, limit=1000)

    # Save Twitter data
    twitter_df = twitter_crawler.save_to_csv("twitter_streaming_data.csv")

    # Combine data and save
    combined_data = []

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

    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv("streaming_opinions_dataset.csv", index=False)
    print(f"Saved combined dataset with {len(combined_df)} records")