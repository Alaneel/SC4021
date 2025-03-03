"""
Solr indexing functionality for the EV opinions search engine
"""
import pandas as pd
import pysolr
import json
import time
import os
import logging
from tqdm import tqdm
import sys

from config.app_config import SOLR_FULL_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("indexer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class SolrIndexer:
    """
    Indexer class for Solr integration
    """

    def __init__(self, solr_url=None):
        """
        Initialize the Solr indexer

        Args:
            solr_url (str): URL to the Solr instance
        """
        self.solr_url = solr_url or SOLR_FULL_URL
        self.solr = pysolr.Solr(self.solr_url, always_commit=False)
        logger.info(f"Initialized Solr indexer with URL: {self.solr_url}")

        # Test connection
        try:
            self.solr.ping()
            logger.info("Successfully connected to Solr")
        except Exception as e:
            logger.error(f"Failed to connect to Solr: {str(e)}")
            logger.error("Make sure Solr is running and the collection is created")

    def preprocess_for_indexing(self, df):
        """
        Preprocess dataframe before indexing to Solr

        Args:
            df (pandas.DataFrame): The dataframe to preprocess

        Returns:
            pandas.DataFrame: Preprocessed dataframe
        """
        # Make a copy to avoid modifying the original
        df = df.copy()

        # Ensure id field exists
        if 'id' not in df.columns:
            logger.info("Generating IDs for documents")
            df['id'] = [f"ev_{i}" for i in range(len(df))]

        # Convert dates to Solr format
        if 'created_utc' in df.columns:
            logger.info("Converting dates to Solr format")
            df['created_utc'] = pd.to_datetime(df['created_utc']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Ensure text fields are strings
        for field in ['text', 'title']:
            if field in df.columns:
                df[field] = df[field].fillna('').astype(str)

        # Handle list fields for Solr
        list_fields = ['topics', 'entities']
        for field in list_fields:
            if field in df.columns:
                df[field] = df[field].apply(self._ensure_list)

        # Remove rows with empty text
        df = df[df['text'].str.strip() != '']

        logger.info(f"Preprocessed {len(df)} documents for indexing")
        return df

    def _ensure_list(self, value):
        """Ensure value is a proper list for Solr"""
        if pd.isna(value) or value is None:
            return []

        if isinstance(value, list):
            return value

        # Try to convert from string representation of list
        if isinstance(value, str):
            try:
                result = eval(value)
                if isinstance(result, list):
                    return result
            except:
                pass

            # Try comma-separated format
            if ',' in value:
                return [item.strip() for item in value.split(',')]

            # Single value
            return [value]

        # Default: wrap in list
        return [str(value)]

    def index_dataframe(self, df, batch_size=500):
        """
        Index a pandas dataframe to Solr

        Args:
            df (pandas.DataFrame): Dataframe to index
            batch_size (int): Batch size for indexing

        Returns:
            int: Number of documents indexed
        """
        # Preprocess the dataframe
        df = self.preprocess_for_indexing(df)

        # Convert to list of dicts for Solr
        documents = df.to_dict('records')
        total_docs = len(documents)

        logger.info(f"Indexing {total_docs} documents to Solr in batches of {batch_size}")

        start_time = time.time()
        indexed_count = 0

        # Process in batches
        for i in tqdm(range(0, total_docs, batch_size), desc="Indexing to Solr"):
            batch = documents[i:i + batch_size]

            try:
                self.solr.add(batch)
                indexed_count += len(batch)

                # Commit every 10 batches or at the end
                if (i + batch_size) >= total_docs or (i // batch_size) % 10 == 0:
                    self.solr.commit()
                    logger.info(f"Committed {indexed_count} documents")

            except Exception as e:
                logger.error(f"Error indexing batch {i // batch_size}: {str(e)}")
                # Try to commit what we have so far
                try:
                    self.solr.commit()
                except:
                    pass

        # Final commit
        try:
            self.solr.commit()
        except Exception as e:
            logger.error(f"Error in final commit: {str(e)}")

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Indexed {indexed_count} documents in {duration:.2f} seconds")
        logger.info(f"Indexing speed: {indexed_count / duration:.2f} docs/second")

        return indexed_count

    def index_csv_file(self, csv_file, batch_size=500):
        """
        Index a CSV file to Solr

        Args:
            csv_file (str): Path to the CSV file
            batch_size (int): Batch size for indexing

        Returns:
            int: Number of documents indexed
        """
        logger.info(f"Loading data from {csv_file}")
        df = pd.read_csv(csv_file)
        return self.index_dataframe(df, batch_size)

    def delete_all_documents(self):
        """
        Delete all documents from the Solr index

        Returns:
            bool: True if successful
        """
        logger.warning("Deleting all documents from Solr index")
        try:
            self.solr.delete(q='*:*')
            self.solr.commit()
            logger.info("Successfully deleted all documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False

    def optimize_index(self):
        """
        Optimize the Solr index for better performance

        Returns:
            bool: True if successful
        """
        logger.info("Optimizing Solr index")
        try:
            self.solr.optimize()
            logger.info("Index optimization complete")
            return True
        except Exception as e:
            logger.error(f"Error optimizing index: {str(e)}")
            return False

    def get_index_stats(self):
        """
        Get statistics about the Solr index

        Returns:
            dict: Index statistics
        """
        logger.info("Retrieving index statistics")
        try:
            # Query for all documents but with rows=0 to just get counts
            response = self.solr.search('*:*', **{
                'rows': 0,
                'facet': 'on',
                'facet.field': ['sentiment', 'subreddit', 'type'],
                'stats': 'true',
                'stats.field': ['sentiment_score', 'score']
            })

            stats = {
                'num_documents': response.hits,
                'facets': response.facets,
                'stats': getattr(response, 'stats', {})
            }

            logger.info(f"Index contains {stats['num_documents']} documents")
            return stats

        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return {'error': str(e)}