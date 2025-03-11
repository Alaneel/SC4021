#!/usr/bin/env python
"""
Script to run the X (Twitter) crawler for collecting EV opinions
"""
import argparse
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler import XCrawler, preprocess_dataset
from config.app_config import (
    X_API_KEY,
    X_API_SECRET,
    X_BEARER_TOKEN,
    X_ACCESS_TOKEN,
    X_ACCESS_SECRET,
    X_USER_AGENT,
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
    parser = argparse.ArgumentParser(description='Run the X (Twitter) EV opinions crawler')
    parser.add_argument('--queries', nargs='+', help='List of search queries (default: use config)')
    parser.add_argument('--limit', type=int, default=100, help='Maximum tweets per query (default: 100)')
    parser.add_argument('--output', help='Output filename (default: auto-generated)')
    parser.add_argument('--bearer-token', help='X Bearer Token (default: from config)')
    parser.add_argument('--api-key', help='X API Key (default: from config)')
    parser.add_argument('--api-secret', help='X API Secret (default: from config)')
    parser.add_argument('--access-token', help='X Access Token (default: from config)')
    parser.add_argument('--access-secret', help='X Access Token Secret (default: from config)')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the collected data')
    parser.add_argument('--force', action='store_true', help='Force crawling even if API credentials are missing')

    args = parser.parse_args()

    # Get API credentials
    bearer_token = args.bearer_token or X_BEARER_TOKEN
    api_key = args.api_key or X_API_KEY
    api_secret = args.api_secret or X_API_SECRET
    access_token = args.access_token or X_ACCESS_TOKEN
    access_secret = args.access_secret or X_ACCESS_SECRET

    # Check if credentials are provided
    if not bearer_token and not args.force:
        logger.error("X Bearer Token not provided. Set environment variables or pass as arguments.")
        logger.error(
            "To use the crawler, you need to create an X Developer account and a project at https://developer.twitter.com/")
        logger.error(
            "Set X_BEARER_TOKEN environment variable, or use --bearer-token argument")
        logger.error("Use --force to attempt to run without credentials (likely to fail)")
        return 1

    # Get queries
    queries = args.queries or SEARCH_QUERIES

    logger.info(f"Starting X crawler with {len(queries)} queries")
    logger.info(f"Limit: {args.limit} tweets per query")

    # Initialize crawler
    crawler = XCrawler(
        api_key=api_key,
        api_secret=api_secret,
        bearer_token=bearer_token,
        access_token=access_token,
        access_secret=access_secret,
        user_agent=X_USER_AGENT
    )

    # Crawl data
    logger.info("Starting crawl process...")
    crawler.search_multiple_queries(queries, max_results_per_query=args.limit)

    # Generate output filename if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or os.path.join(RAW_DATA_DIR, f"x_ev_opinions_{timestamp}.csv")

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