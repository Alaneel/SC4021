#!/usr/bin/env python
"""
Integration test for mock crawler with classification and indexing
"""
import os
import sys
import logging
import tempfile
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from crawler.mock_crawler import MockNewsAPICrawler
from crawler.data_cleaner import preprocess_dataset
from classification import EVOpinionClassifier
from indexing import SolrIndexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_integration_test(num_articles=100, test_solr_indexing=False):
    """
    Run an integration test for the mock crawler with processing pipeline

    Args:
        num_articles (int): Number of articles to generate
        test_solr_indexing (bool): Whether to test Solr indexing

    Returns:
        bool: True if successful
    """
    logger.info(f"Running integration test with {num_articles} mock articles")

    # Step 1: Generate mock data
    logger.info("Step 1: Generating mock data...")
    crawler = MockNewsAPICrawler()
    crawler.generate_mock_data(num_articles)

    # Create a temporary directory for test outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = os.path.join(temp_dir, f"mock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        # Step 2: Save mock data to CSV
        logger.info(f"Step 2: Saving mock data to {csv_path}...")
        df = crawler.save_to_csv(csv_path)

        # Step 3: Preprocess the data
        logger.info("Step 3: Preprocessing mock data...")
        processed_df = preprocess_dataset(df)
        processed_path = os.path.join(temp_dir, "processed_mock_data.csv")
        processed_df.to_csv(processed_path, index=False)

        # Step 4: Run classification
        logger.info("Step 4: Classifying mock data...")
        classifier = EVOpinionClassifier()
        classifier.load_sentiment_model()

        # Process a subset for faster testing
        test_subset = processed_df.head(min(20, len(processed_df)))
        classified_df = classifier.process_data(
            test_subset,
            analyze_sentiment=True,
            extract_entities=True,
            assign_topics=False  # Skip topic modeling to speed up the test
        )

        # Verify classification results
        logger.info("Verification: Sentiment classification results")
        sentiment_counts = classified_df['sentiment'].value_counts()
        logger.info(f"Sentiment distribution: {sentiment_counts.to_dict()}")

        entity_counts = {}
        for entities in classified_df['entities']:
            if isinstance(entities, list):
                for entity in entities:
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1

        logger.info(f"Top entities detected: {sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")

        # Step 5: Test Solr indexing (optional)
        if test_solr_indexing:
            try:
                logger.info("Step 5: Testing Solr indexing...")
                indexer = SolrIndexer()

                # Only index a small sample
                index_sample = classified_df.head(10)
                num_indexed = indexer.index_dataframe(index_sample, batch_size=10)

                logger.info(f"Successfully indexed {num_indexed} documents to Solr")

                # Test search
                from indexing.search_utils import execute_search
                search_results = execute_search('*:*', start=0, rows=5)
                logger.info(f"Search test found {search_results.get('num_found', 0)} documents")

            except Exception as e:
                logger.warning(f"Solr indexing test failed: {str(e)}")
                logger.warning("This is expected if Solr is not running")

        logger.info("Integration test completed successfully!")
        return True


if __name__ == "__main__":
    # Default to not testing Solr indexing since it requires Solr to be running
    test_solr = '--test-solr' in sys.argv
    run_integration_test(num_articles=50, test_solr_indexing=test_solr)