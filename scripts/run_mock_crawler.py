#!/usr/bin/env python
"""
Script to run the Mock News API crawler for generating synthetic EV opinions
"""
import argparse
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import mock crawler
from crawler.mock_crawler import MockNewsAPICrawler

# Import project modules
from crawler.data_cleaner import preprocess_dataset
from config.app_config import (
    SEARCH_QUERIES,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    NEWSAPI_DAYS_BACK,
    NEWSAPI_MAX_RESULTS_PER_QUERY
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run the Mock News API crawler for EV opinions')
    parser.add_argument('--num-articles', type=int, default=1000,
                        help='Number of articles to generate (default: 1000)')
    parser.add_argument('--queries', nargs='+', help='List of search queries (default: use config)')
    parser.add_argument('--output', help='Output filename (default: auto-generated)')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess the generated data')
    parser.add_argument('--use-queries', action='store_true',
                        help='Generate data using search_multiple_queries method instead of direct generation')

    args = parser.parse_args()

    # Initialize crawler
    crawler = MockNewsAPICrawler()

    # Generate data
    if args.use_queries:
        # Use the search_multiple_queries method to mimic the actual crawler behavior
        queries = args.queries or SEARCH_QUERIES
        logger.info(f"Generating mock data using {len(queries)} search queries...")

        # Calculate suitable number of results per query
        results_per_query = args.num_articles // len(queries)
        if results_per_query < 1:
            results_per_query = 1

        crawler.search_multiple_queries(
            queries=queries,
            max_results_per_query=results_per_query,
            days_back=NEWSAPI_DAYS_BACK
        )
    else:
        # Generate data directly
        logger.info(f"Generating {args.num_articles} mock articles...")
        crawler.generate_mock_data(args.num_articles)

    # Generate output filename if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or os.path.join(RAW_DATA_DIR, f"mock_news_ev_opinions_{timestamp}.csv")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save generated data
    df = crawler.save_to_csv(output_file)

    if df is None or len(df) == 0:
        logger.warning("No data was generated.")
        return 0

    # Preprocess if requested
    if args.preprocess and df is not None:
        logger.info("Preprocessing generated data...")
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