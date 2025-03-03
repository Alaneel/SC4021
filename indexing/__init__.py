"""
Indexing module for Solr integration
"""
from .solr_indexer import SolrIndexer
from .search_utils import execute_search, format_search_results

__all__ = ['SolrIndexer', 'execute_search', 'format_search_results']