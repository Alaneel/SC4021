"""
News API crawler for collecting Electric Vehicle opinions from news sources
"""
import requests
import pandas as pd
import time
import random
from datetime import datetime, timedelta
import os
import logging
from tqdm import tqdm
import sys
import re
import json
from urllib.parse import quote

from .data_cleaner import clean_text
from config.app_config import (
    NEWSAPI_API_KEY,
    NEWSAPI_ENDPOINT,
    SEARCH_QUERIES,
    RAW_DATA_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class NewsAPICrawler:
    """
    Crawler for collecting EV-related opinions from news sources using News API
    """

    def __init__(self, api_key=None):
        """
        Initialize the News API crawler

        Args:
            api_key (str): News API key
        """
        self.api_key = api_key or NEWSAPI_API_KEY
        self.endpoint = NEWSAPI_ENDPOINT

        if not self.api_key:
            raise ValueError("News API Key not provided. "
                             "Set NEWSAPI_API_KEY environment variable "
                             "or pass it as a parameter.")

        self.data = []
        logger.info("News API crawler initialized")

    def _process_article(self, article, search_query=None):
        """
        Process an article into a standardized format

        Args:
            article: Article object from News API
            search_query (str): The search query that found this article

        Returns:
            dict: Processed article data
        """
        # Extract article publication date
        published_at = article.get('publishedAt')
        if published_at:
            try:
                pub_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
                formatted_date = pub_date.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                formatted_date = published_at
        else:
            formatted_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Generate a unique ID for the article
        article_id = f"news_{article.get('source', {}).get('id', 'unknown')}_{int(time.time())}_{random.randint(1000, 9999)}"

        # Extract article content
        content = article.get('content', '')
        if not content:
            content = article.get('description', '')

        # Create a score based on relevance or popularity metrics
        # For News API, we don't have direct score, so we'll create a synthetic one
        # based on freshness and source reputation
        days_old = (datetime.now() - pub_date).days if published_at else 30
        score = max(1, 100 - (days_old * 3))  # Newer articles get higher scores

        # Extract article data
        article_data = {
            'id': article_id,
            'type': 'news',
            'title': article.get('title', ''),
            'text': content,
            'author': article.get('author', 'Unknown'),
            'created_utc': formatted_date,
            'score': score,
            'subreddit': 'news',  # Use 'news' as a platform indicator
            'url': article.get('url', ''),
            'source_name': article.get('source', {}).get('name', 'Unknown Source'),
            'source_id': article.get('source', {}).get('id', 'unknown'),
            'description': article.get('description', ''),
            'search_query': search_query or '',
            'platform': 'news'
        }

        return article_data

    def search_articles(self, query, days_back=30, max_results=100, language='en'):
        """
        Search News API for articles matching a query

        Args:
            query (str): Search query
            days_back (int): Number of days to look back
            max_results (int): Maximum number of articles to retrieve
            language (str): Language of articles to retrieve

        Returns:
            int: Number of new items collected
        """
        logger.info(f"Searching News API for: '{query}'...")
        initial_count = len(self.data)

        try:
            # Set fixed date range that works with the free tier
            # Setting the end date to February 9, 2025 (one day before the limit)
            end_date = datetime(2025, 2, 9)
            start_date = end_date - timedelta(days=days_back)

            # Format dates for API
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')

            # Encode query for URL
            encoded_query = quote(query)

            # Set up pagination
            collected = 0
            page = 1
            page_size = min(100, max_results)  # News API allows max 100 results per request

            # News API search pagination loop
            with tqdm(total=max_results, desc=f"Collecting articles for '{query}'") as pbar:
                while collected < max_results:
                    # Construct request URL with parameters
                    url = f"{self.endpoint}?q={encoded_query}&from={from_date}&to={to_date}&language={language}&sortBy=relevancy&pageSize={page_size}&page={page}&apiKey={self.api_key}"

                    try:
                        # Execute search with pagination
                        response = requests.get(url)
                        response.raise_for_status()  # Raise exception for error status codes
                        data = response.json()

                        # Check for API errors
                        if data.get('status') != 'ok':
                            error_msg = data.get('message', 'Unknown error')
                            logger.error(f"News API error: {error_msg}")
                            break

                        # Get articles from response
                        articles = data.get('articles', [])

                        # Check if we got results
                        if not articles:
                            logger.info(f"No more results for query: '{query}'")
                            break

                        # Process articles
                        for article in articles:
                            # Process article
                            article_data = self._process_article(article, query)

                            # Check for duplicates by URL
                            if not any(item['url'] == article_data['url'] for item in self.data):
                                # Add to collection
                                self.data.append(article_data)
                                collected += 1
                                pbar.update(1)

                                # Check if we've reached the limit
                                if collected >= max_results:
                                    break

                        # Check if we've reached the last page
                        total_results = data.get('totalResults', 0)
                        total_pages = (total_results + page_size - 1) // page_size

                        if page >= total_pages:
                            logger.info(f"Reached last page for query: '{query}'")
                            break

                        # Move to next page
                        page += 1

                        # Add delay to respect rate limits (News API allows 100 requests per day on free tier)
                        time.sleep(1)

                    except requests.exceptions.RequestException as e:
                        logger.error(f"Request error: {str(e)}")
                        time.sleep(5)  # Wait before retrying
                        break
                    except json.decoder.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {str(e)}")
                        time.sleep(5)
                        break
                    except Exception as e:
                        logger.error(f"Error searching articles: {str(e)}")
                        time.sleep(5)
                        break

            new_items = len(self.data) - initial_count
            logger.info(f"Collected {new_items} new articles for '{query}'")
            return new_items

        except Exception as e:
            logger.error(f"Error in search_articles: {str(e)}")
            return 0

    def search_multiple_queries(self, queries, max_results_per_query=100, days_back=30):
        """
        Search News API for multiple queries

        Args:
            queries (list): List of search queries
            max_results_per_query (int): Maximum articles per query
            days_back (int): Number of days to look back

        Returns:
            int: Total number of items collected
        """
        initial_count = len(self.data)

        # Track progress with tqdm
        total_queries = len(queries)
        with tqdm(total=total_queries, desc="Searching News API") as pbar:
            for query in queries:
                self.search_articles(query, days_back=days_back, max_results=max_results_per_query)
                pbar.update(1)

                # Save intermediate results every 5 iterations
                if len(self.data) % 500 == 0 and len(self.data) > 0:
                    self.save_intermediate_results()

                # Add a delay between queries to respect rate limits
                delay = 2 + random.uniform(1, 3)  # 3-5 seconds between queries
                time.sleep(delay)

        new_items = len(self.data) - initial_count
        logger.info(f"Total new items collected: {new_items}")
        return new_items

    def save_intermediate_results(self):
        """Save intermediate results to avoid data loss"""
        if not self.data:
            return

        # Create timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RAW_DATA_DIR, f"news_ev_opinions_intermediate_{timestamp}.csv")

        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} records to {filename}")

    def save_to_csv(self, filename=None):
        """
        Save the collected data to a CSV file

        Args:
            filename (str): Output filename

        Returns:
            pandas.DataFrame: The saved dataframe
        """
        if not self.data:
            logger.warning("No data to save")
            return None

        if filename is None:
            filename = os.path.join(RAW_DATA_DIR, f"news_ev_opinions_{datetime.now().strftime('%Y%m%d')}.csv")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        df = pd.DataFrame(self.data)

        # Remove duplicates by URL
        df = df.drop_duplicates(subset=['url'])

        # Save to CSV
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} records to {filename}")

        # Print some statistics
        total_records = len(df)
        logger.info(f"Total records: {total_records}")

        # Word count stats
        df['word_count'] = df['text'].str.split().str.len()
        total_words = df['word_count'].sum()
        unique_words = len(set(' '.join(df['text'].dropna()).split()))

        logger.info(f"Total words: {total_words}")
        logger.info(f"Unique words: {unique_words}")
        logger.info(f"Average words per record: {total_words / total_records:.1f}")

        return df