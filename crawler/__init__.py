"""
Crawler module for collecting EV opinions from news sources
"""
from .newsapi_crawler import NewsAPICrawler
from .data_cleaner import clean_text, preprocess_dataset, extract_article_content, extract_entities

__all__ = ['NewsAPICrawler', 'clean_text', 'preprocess_dataset', 'extract_article_content', 'extract_entities']

# Version tracking for the crawler module
__version__ = '1.1.0'  # Updated for fixed date range support