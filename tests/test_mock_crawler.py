#!/usr/bin/env python
"""
Test script for the Mock News API crawler
"""
import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import mock crawler
from crawler.mock_crawler import MockNewsAPICrawler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_mock_crawler():
    """Test the mock crawler functionality"""
    logger.info("Testing Mock News API crawler...")

    # Initialize crawler
    crawler = MockNewsAPICrawler()

    # Generate a small sample of articles
    num_articles = 100
    logger.info(f"Generating {num_articles} mock articles...")
    crawler.generate_mock_data(num_articles)

    # Check the number of articles generated
    assert len(crawler.data) == num_articles, f"Expected {num_articles} articles, got {len(crawler.data)}"

    # Save to a temporary CSV file
    temp_dir = "test_output"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, "test_mock_articles.csv")
    df = crawler.save_to_csv(temp_file)

    # Check if the file was created
    assert os.path.exists(temp_file), f"CSV file {temp_file} not created"

    # Verify the dataframe has the correct columns
    required_columns = ['id', 'type', 'title', 'text', 'author', 'created_utc',
                        'source_name', 'url', 'platform']
    for col in required_columns:
        assert col in df.columns, f"Column {col} missing from dataframe"

    # Analyze the content
    logger.info("Analyzing mock content...")

    # Brands mentioned
    brands_mentioned = []
    for text in df['text']:
        for brand in crawler.EV_BRANDS:
            if brand.lower() in text.lower():
                brands_mentioned.append(brand)

    brand_counts = Counter(brands_mentioned)
    logger.info(f"Top brands mentioned: {brand_counts.most_common(5)}")

    # Word count distribution
    df['word_count'] = df['text'].str.split().str.len()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'])
    plt.title('Word Count Distribution in Mock Articles')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(temp_dir, 'word_count_distribution.png'))
    logger.info(f"Saved word count distribution to {os.path.join(temp_dir, 'word_count_distribution.png')}")

    # Date distribution
    df['date'] = pd.to_datetime(df['created_utc']).dt.date

    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='date', order=sorted(df['date'].unique()))
    plt.title('Date Distribution of Mock Articles')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, 'date_distribution.png'))
    logger.info(f"Saved date distribution to {os.path.join(temp_dir, 'date_distribution.png')}")

    # Source distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='source_name')
    plt.title('Distribution of News Sources')
    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, 'source_distribution.png'))
    logger.info(f"Saved source distribution to {os.path.join(temp_dir, 'source_distribution.png')}")

    logger.info("All tests passed successfully!")
    logger.info(f"Test output saved to {temp_dir} directory")

    return df


if __name__ == "__main__":
    test_mock_crawler()