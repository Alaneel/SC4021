#!/usr/bin/env python
"""
Script to run the News API crawler for collecting EV opinions
"""
import argparse
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler import NewsAPICrawler, preprocess_dataset
from config.app_config import (
    NEWSAPI_API_KEY,
    SEARCH_QUERIES,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    NEWSAPI_DAYS_BACK,
    NEWSAPI_MAX_RESULTS_PER_QUERY,
    NEWSAPI_LANGUAGE,
    NEWSAPI_END_DATE,
    NEWSAPI_START_DATE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run the News API EV opinions crawler')
    parser.add_argument('--queries', nargs='+', help='List of search queries (default: use config)')
    parser.add_argument('--limit', type=int, default=NEWSAPI_MAX_RESULTS_PER_QUERY,
                        help=f'Maximum articles per query (default: {NEWSAPI_MAX_RESULTS_PER_QUERY})')
    parser.add_argument('--days', type=int, default=NEWSAPI_DAYS_BACK,
                        help=f'Number of days to look back (default: {NEWSAPI_DAYS_BACK})')
    parser.add_argument('--language', default=NEWSAPI_LANGUAGE,
                        help=f'Language of articles (default: {NEWSAPI_LANGUAGE})')
    parser.add_argument('--output', help='Output filename (default: auto-generated)')
    parser.add_argument('--api-key', help='News API Key (default: from config)')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the collected data')
    parser.add_argument('--force', action='store_true', help='Force crawling even if API credentials are missing')

    args = parser.parse_args()

    # Get API credentials
    api_key = args.api_key or NEWSAPI_API_KEY

    # Check if credentials are provided
    if not api_key and not args.force:
        logger.error("News API Key not provided. Set environment variables or pass as arguments.")
        logger.error(
            "To use the crawler, you need to create a News API account at https://newsapi.org/")
        logger.error(
            "Set NEWSAPI_API_KEY environment variable, or use --api-key argument")
        logger.error("Use --force to attempt to run without credentials (likely to fail)")
        return 1

    # Get queries
    queries = args.queries or SEARCH_QUERIES

    logger.info(f"Starting News API crawler with {len(queries)} queries")
    logger.info(f"Limit: {args.limit} articles per query, looking back {args.days} days")

    # Initialize crawler
    crawler = NewsAPICrawler(api_key=api_key)

    # Crawl data
    logger.info("Starting crawl process...")
    crawler.search_multiple_queries(queries, max_results_per_query=args.limit, days_back=args.days)

    # Generate output filename if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or os.path.join(RAW_DATA_DIR, f"news_ev_opinions_{timestamp}.csv")

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