# main.py
"""Main script to run the social media crawlers."""

from crawler import RedditCrawler, TwitterCrawler, combine_datasets
from credentials import (
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET,
    TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET,
    TWITTER_BEARER_TOKEN
)
from config import (
    DEFAULT_REDDIT_LIMIT,
    DEFAULT_TWITTER_MAX_RESULTS, DEFAULT_TWITTER_LIMIT,
    SUBREDDITS, TWITTER_QUERIES,
    REDDIT_OUTPUT_CSV, TWITTER_OUTPUT_CSV, COMBINED_OUTPUT_CSV
)


def main():
    """Main function to execute the crawling process."""
    # Initialize Reddit crawler
    reddit_crawler = RedditCrawler(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    # Crawl multiple subreddits
    for subreddit in SUBREDDITS:
        reddit_crawler.crawl_subreddit(subreddit, limit=DEFAULT_REDDIT_LIMIT)

    # Save Reddit data
    reddit_df = reddit_crawler.save_to_csv(REDDIT_OUTPUT_CSV)

    # Twitter crawler (commented out as in original code)
    """
    # Initialize Twitter crawler
    twitter_crawler = TwitterCrawler(
        consumer_key=TWITTER_CONSUMER_KEY,
        consumer_secret=TWITTER_CONSUMER_SECRET,
        access_token=TWITTER_ACCESS_TOKEN,
        access_token_secret=TWITTER_ACCESS_TOKEN_SECRET,
        bearer_token=TWITTER_BEARER_TOKEN
    )

    # Crawl tweets for different streaming services
    for query in TWITTER_QUERIES:
        twitter_crawler.crawl_tweets(
            query=query,
            max_results=DEFAULT_TWITTER_MAX_RESULTS,
            limit=DEFAULT_TWITTER_LIMIT
        )

    # Save Twitter data
    twitter_df = twitter_crawler.save_to_csv(TWITTER_OUTPUT_CSV)

    # Combine Reddit and Twitter data
    combine_datasets(reddit_df, twitter_df, COMBINED_OUTPUT_CSV)
    """

    # For now, just save Reddit data as the combined dataset (as in original code)
    combine_datasets(reddit_df, output_filename=COMBINED_OUTPUT_CSV)


if __name__ == "__main__":
    main()