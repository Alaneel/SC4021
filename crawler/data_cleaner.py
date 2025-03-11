"""
Text cleaning utilities for preprocessing news article data
"""
import re
import pandas as pd
import logging
import html
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


def clean_text(text):
    """
    Clean the text by removing HTML, excessive whitespace, etc.

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

    # Remove truncation artifacts from News API
    text = re.sub(r'\[\+\d+ chars\]$', '', text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags using BeautifulSoup
    try:
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
    except:
        # Fallback to regex if BeautifulSoup fails
        text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'https\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove excessive punctuation
    text = re.sub(r'([.!?]){2,}', r'\1', text)

    # Remove special characters but keep sentence structure
    text = re.sub(r'[^\w\s.,!?;:()\-\']', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_article_content(row):
    """
    Extract the most relevant content from an article by combining title,
    description, and content fields.

    Args:
        row (pandas.Series): A row from the articles DataFrame

    Returns:
        str: Combined and cleaned article content
    """
    # Get title, description, and content
    title = row.get('title', '')
    description = row.get('description', '')
    content = row.get('text', '')

    # Clean each part
    clean_title = clean_text(title)
    clean_description = clean_text(description)
    clean_content = clean_text(content)

    # Combine parts with preference for longer content
    if len(clean_content) > 50:
        full_text = clean_content
    elif len(clean_description) > 30:
        full_text = clean_description
    else:
        full_text = clean_title

    # Add title at the beginning if not already included
    if clean_title and clean_title not in full_text:
        full_text = clean_title + ". " + full_text

    return full_text


def remove_duplicates(df):
    """
    Remove duplicate content from the dataframe.

    Args:
        df (pandas.DataFrame): DataFrame containing the articles

    Returns:
        pandas.DataFrame: DataFrame with duplicates removed
    """
    logger.info(f"Original dataset size: {len(df)}")

    # Drop exact duplicates by URL
    df = df.drop_duplicates(subset=['url'])
    logger.info(f"After dropping URL duplicates: {len(df)}")

    # Check for near-duplicates in title/content
    # Simplified approach: Check by title similarity
    df['title_lower'] = df['title'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

    # Sort by score (higher first) and created_utc (newer first)
    df = df.sort_values(['score', 'created_utc'], ascending=[False, False])

    # Group by similar titles and keep the highest scored/newest
    df = df.drop_duplicates(subset=['title_lower'])

    # Drop temporary columns
    df = df.drop(columns=['title_lower'])

    logger.info(f"After dropping near-duplicates: {len(df)}")
    return df


def extract_entities(text, ev_keywords):
    """
    Extract EV-related entities from text using keyword matching

    Args:
        text (str): Input text
        ev_keywords (dict): Dictionary of EV keywords

    Returns:
        list: Extracted entities
    """
    if not text or pd.isna(text):
        return []

    entities = []
    text_lower = text.lower()

    # Check for EV entities in the text
    for brand, models in ev_keywords.items():
        for entity in models:
            if entity in text_lower:
                entities.append(entity)
                # Also add the brand if we found a model
                if entity != brand and brand not in entities:
                    entities.append(brand)

    # Add general EV terminology
    ev_terms = ["ev", "electric vehicle", "electric car", "battery electric",
                "charging", "charger", "range", "battery"]

    for term in ev_terms:
        if term in text_lower:
            entities.append(term)

    return list(set(entities))  # Remove duplicates


def preprocess_dataset(df):
    """
    Preprocess the dataset by cleaning text and removing duplicates

    Args:
        df (pandas.DataFrame): Original DataFrame

    Returns:
        pandas.DataFrame: Processed DataFrame
    """
    logger.info("Preprocessing dataset...")

    # Define EV keywords for entity extraction
    ev_keywords = {
        "tesla": ["tesla", "model s", "model 3", "model y", "model x", "cybertruck"],
        "rivian": ["rivian", "r1t", "r1s"],
        "lucid": ["lucid", "air"],
        "gm": ["chevrolet bolt", "bolt ev", "bolt euv", "chevy bolt", "gm"],
        "ford": ["ford", "mustang mach e", "mach e", "f-150 lightning"],
        "nissan": ["nissan leaf", "leaf", "ariya"],
        "volkswagen": ["volkswagen", "vw", "id.4", "id4"],
        "hyundai": ["hyundai", "ioniq", "kona"],
        "kia": ["kia", "ev6", "niro"],
        "audi": ["audi", "e-tron"],
        "porsche": ["porsche", "taycan"],
        "bmw": ["bmw", "i3", "i4", "ix"]
    }

    # Extract and clean text content
    logger.info("Extracting and cleaning article content...")
    df['text'] = df.apply(extract_article_content, axis=1)

    # Clean title field
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

    # Extract entities
    logger.info("Extracting EV-related entities...")
    df['entities'] = df['text'].apply(lambda x: extract_entities(x, ev_keywords))

    # Assign platform
    df['platform'] = 'news'

    logger.info(f"Final preprocessed dataset size: {len(df)}")
    return df