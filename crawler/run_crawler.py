# main.py
"""Main script to run the Reddit crawler."""

from crawler import RedditCrawler, combine_datasets
from credentials import (
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
)
from config import (
    DEFAULT_REDDIT_LIMIT,
    SUBREDDITS,
    REDDIT_OUTPUT_CSV,
    COMBINED_OUTPUT_CSV
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

    # Save Reddit data as the combined dataset
    combine_datasets(reddit_df, output_filename=COMBINED_OUTPUT_CSV)


if __name__ == "__main__":
    main()