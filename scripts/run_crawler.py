#!/usr/bin/env python
"""
Script to run the Reddit crawler for collecting EV opinions
"""
import argparse
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler import RedditCrawler, preprocess_dataset
from config.app_config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    SUBREDDITS,
    SEARCH_QUERIES,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run the Reddit EV opinions crawler')
    parser.add_argument('--subreddits', nargs='+', help='List of subreddits to crawl (default: use config)')
    parser.add_argument('--queries', nargs='+', help='List of search queries (default: use config)')
    parser.add_argument('--limit', type=int, default=100, help='Maximum submissions per query (default: 100)')
    parser.add_argument('--output', help='Output filename (default: auto-generated)')
    parser.add_argument('--client-id', help='Reddit API client ID (default: from config)')
    parser.add_argument('--client-secret', help='Reddit API client secret (default: from config)')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the collected data')
    parser.add_argument('--force', action='store_true', help='Force crawling even if API credentials are missing')

    args = parser.parse_args()

    # Get API credentials
    client_id = args.client_id or REDDIT_CLIENT_ID
    client_secret = args.client_secret or REDDIT_CLIENT_SECRET

    # Check if credentials are provided
    if (not client_id or not client_secret) and not args.force:
        logger.error("Reddit API credentials not provided. Set environment variables or pass as arguments.")
        logger.error(
            "To use the crawler, you need to create a Reddit API application at https://www.reddit.com/prefs/apps")
        logger.error(
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables, or use --client-id and --client-secret arguments")
        logger.error("Use --force to attempt to run without credentials (likely to fail)")
        return 1

    # Get subreddits and queries
    subreddits = args.subreddits or SUBREDDITS
    queries = args.queries or SEARCH_QUERIES

    logger.info(f"Starting Reddit crawler with {len(subreddits)} subreddits and {len(queries)} queries")
    logger.info(f"Limit: {args.limit} submissions per query")

    # Initialize crawler
    crawler = RedditCrawler(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=REDDIT_USER_AGENT
    )

    # Crawl data
    logger.info("Starting crawl process...")
    crawler.crawl_multiple(subreddits, queries, limit_per_query=args.limit)

    # Generate output filename if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or os.path.join(RAW_DATA_DIR, f"reddit_ev_opinions_{timestamp}.csv")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save collected data
    df = crawler.save_to_csv(output_file)

    if df is None or len(df) == 0:
        logger.warning("No data was collected.")
        return 0

    # Preprocess if requested
    if args.preprocess and df is not None:
        logger.info("Preprocessing collected data...")
        processed_df = preprocess_dataset(df)

        # Save processed data
        processed_file = os.path.join(PROCESSED_DATA_DIR, f"processed_ev_opinions_{timestamp}.csv")
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
        processed_df.to_csv(processed_file, index=False)

        logger.info(f"Saved processed data to {processed_file}")
        logger.info(f"Original records: {len(df)}, Processed records: {len(processed_df)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())