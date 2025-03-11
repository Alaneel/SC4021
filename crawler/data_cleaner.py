"""
Text cleaning utilities for preprocessing X (Twitter) data
"""
import re
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def clean_text(text):
    """
    Clean the text by removing URLs, special characters, etc.

    Args:
        text (str): The input text to clean

    Returns:
        str: Cleaned text
    """
    if text is None or pd.isna(text):
        return ""

    # Convert to string if necessary
    if not isinstance(text, str):
        text = str(text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'https\S+', '', text)

    # Convert X-specific entities to more readable form
    # Replace @mentions with "User: "
    text = re.sub(r'@(\w+)', r'User: \1', text)

    # Replace #hashtags with just the word
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove RT prefix commonly found in retweets
    text = re.sub(r'^RT\s+', '', text)

    # Remove markdown formatting
    text = re.sub(r'\*\*|\*|~~|==|__|>!|!<|>|#|`', '', text)

    # Remove special characters but keep sentence structure
    text = re.sub(r'[^\w\s.,!?;:()\-\']', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_duplicates(df):
    """
    Remove duplicate content from the dataframe.

    Args:
        df (pandas.DataFrame): DataFrame containing the opinions

    Returns:
        pandas.DataFrame: DataFrame with duplicates removed
    """
    logger.info(f"Original dataset size: {len(df)}")

    # Drop exact duplicates
    df = df.drop_duplicates(subset=['text'])
    logger.info(f"After dropping exact duplicates: {len(df)}")

    # Check for near-duplicates (e.g., minor variations)
    # Simplified approach: Check by length and first/last 20 characters
    df['text_len'] = df['text'].str.len()
    df['text_start'] = df['text'].str[:20]
    df['text_end'] = df['text'].str[-20:]

    # Sort by score (higher first) and created_utc (newer first)
    df = df.sort_values(['score', 'created_utc'], ascending=[False, False])

    # Group by similar content and keep the highest scored/newest
    df = df.drop_duplicates(subset=['text_len', 'text_start', 'text_end'])

    # Drop temporary columns
    df = df.drop(columns=['text_len', 'text_start', 'text_end'])

    logger.info(f"After dropping near-duplicates: {len(df)}")
    return df


def preprocess_dataset(df):
    """
    Preprocess the dataset by cleaning text and removing duplicates

    Args:
        df (pandas.DataFrame): Original DataFrame

    Returns:
        pandas.DataFrame: Processed DataFrame
    """
    logger.info("Preprocessing dataset...")

    # Clean text fields
    df['text'] = df['text'].apply(clean_text)

    if 'title' in df.columns:
        df['title'] = df['title'].apply(clean_text)

    # Remove rows with empty text
    df = df[df['text'].str.len() > 20]
    logger.info(f"After removing short texts: {len(df)}")

    # Remove duplicates
    df = remove_duplicates(df)

    # Add word count
    df['word_count'] = df['text'].str.split().str.len()

    # Filter out very short content
    df = df[df['word_count'] >= 5]

    # Process X-specific fields
    if 'hashtags' in df.columns:
        # Convert comma-separated hashtags to list format for Solr indexing
        df['hashtags'] = df['hashtags'].apply(lambda x: x.split(',') if isinstance(x, str) and x else [])

    if 'mentions' in df.columns:
        # Convert comma-separated mentions to list format for Solr indexing
        df['mentions'] = df['mentions'].apply(lambda x: x.split(',') if isinstance(x, str) and x else [])

    logger.info(f"Final preprocessed dataset size: {len(df)}")
    return df