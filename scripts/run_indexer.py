#!/usr/bin/env python
"""
Script to run the Solr indexer for EV opinions
"""
import argparse
import os
import sys
import logging
import pandas as pd
import glob

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing import SolrIndexer
from classification import EVOpinionClassifier
from config.app_config import (
    SOLR_FULL_URL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    MODELS_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run the Solr indexer for EV opinions')
    parser.add_argument('--input', help='Input CSV file path (required if not using --latest)')
    parser.add_argument('--latest', action='store_true', help='Use the latest data file from processed directory')
    parser.add_argument('--solr-url', default=SOLR_FULL_URL, help=f'Solr URL (default: {SOLR_FULL_URL})')
    parser.add_argument('--batch-size', type=int, default=500, help='Batch size for indexing (default: 500)')
    parser.add_argument('--classify', action='store_true', help='Run classification before indexing')
    parser.add_argument('--model-path', help='Path to pre-trained models (default: auto-detect)')
    parser.add_argument('--clear', action='store_true', help='Clear existing index before indexing')
    parser.add_argument('--optimize', action='store_true', help='Optimize the index after indexing')

    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.latest:
        logger.error("Either --input or --latest must be specified")
        return 1

    # Find the latest file if requested
    if args.latest:
        # Check processed directory first
        processed_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "processed_*.csv"))
        raw_files = glob.glob(os.path.join(RAW_DATA_DIR, "reddit_*.csv"))

        if processed_files:
            # Sort by modification time (newest first)
            input_file = max(processed_files, key=os.path.getmtime)
            logger.info(f"Using latest processed file: {input_file}")
        elif raw_files:
            # Sort by modification time (newest first)
            input_file = max(raw_files, key=os.path.getmtime)
            logger.info(f"Using latest raw file: {input_file}")
        else:
            logger.error("No data files found in processed or raw directories")
            return 1
    else:
        input_file = args.input
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return 1

    # Initialize Solr indexer
    indexer = SolrIndexer(solr_url=args.solr_url)

    # Clear index if requested
    if args.clear:
        logger.info("Clearing existing index...")
        indexer.delete_all_documents()

    # Load data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} records")

    # Run classification if requested
    if args.classify:
        logger.info("Running classification on the data before indexing")

        # Initialize classifier
        classifier = EVOpinionClassifier()

        # Try to load pre-trained models
        model_path = args.model_path
        if not model_path:
            # Auto-detect models in the models directory
            model_files = glob.glob(os.path.join(MODELS_DIR, "ev_classifier_*.model"))
            if model_files:
                model_path = os.path.join(MODELS_DIR, "ev_classifier")
                logger.info(f"Auto-detected model files at {model_path}")
                classifier.load_models(model_path)
            else:
                logger.warning("No pre-trained models found, will use default models")
        else:
            classifier.load_models(model_path)

        # Process the data
        logger.info("Processing data with classification pipeline...")
        df = classifier.process_data(
            df,
            analyze_sentiment=True,
            extract_entities=True,
            assign_topics=True
        )

    # Index the data
    logger.info(f"Indexing data to Solr: {args.solr_url}")
    num_indexed = indexer.index_dataframe(df, batch_size=args.batch_size)

    # Optimize if requested
    if args.optimize:
        logger.info("Optimizing the index...")
        indexer.optimize_index()

    # Display statistics
    logger.info(f"Indexing complete! Indexed {num_indexed} documents")

    # Get index stats
    stats = indexer.get_index_stats()
    if 'num_documents' in stats:
        logger.info(f"Total documents in index: {stats['num_documents']}")

    if 'facets' in stats and 'facet_fields' in stats['facets']:
        if 'sentiment' in stats['facets']['facet_fields']:
            sentiment_counts = {}
            for i in range(0, len(stats['facets']['facet_fields']['sentiment']), 2):
                sentiment = stats['facets']['facet_fields']['sentiment'][i]
                count = stats['facets']['facet_fields']['sentiment'][i + 1]
                sentiment_counts[sentiment] = count

            logger.info("Sentiment distribution:")
            for sentiment, count in sentiment_counts.items():
                logger.info(f"  - {sentiment}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())