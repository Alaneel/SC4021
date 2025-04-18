import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import pipeline
import warnings
import spacy
nlp = spacy.load("en_core_web_sm")
import string
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import numpy as np
from scipy.special import softmax

# Load the irony detection model
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Test the model with a sample input
    test_text = "Oh great, just what I needed."
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[0].detach().cpu().numpy()
        scores = softmax(scores)
        prediction = np.argmax(scores)
        confidence = scores[prediction]
        print(f"Sarcasm model test output: Prediction={prediction}, Confidence={confidence:.4f}")

    print("Sarcasm model loaded and validated.")

except Exception as e:
    print("Sarcasm model could not be loaded or validated:", e)
    model = None

# Batch sarcasm detection function
def detect_sarcasm_batch(text_list, batch_size=16):
    if model is None or not text_list:
        print("Sarcasm detector unavailable — returning 0s.")
        return [0] * len(text_list), [0.0] * len(text_list)

    predictions = []
    confidences = []

    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            batch_preds = torch.argmax(probs, dim=1).cpu().tolist()
            batch_confs = probs.max(dim=1).values.cpu().tolist()

            predictions.extend(batch_preds)
            confidences.extend(batch_confs)

    return predictions, confidences


# Word Sense Disambiguation using NLTK's Lesk algorithm
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

def apply_wsd(text):
    tokens = word_tokenize(text)
    senses = [lesk(tokens, word) for word in tokens]
    return " ".join([sense.name() if sense else word for sense, word in zip(senses, tokens)])

# Named Entity Recognition (NER)
def extract_named_entity_labels(text):
    doc = nlp(text)
    labels = [ent.label_ for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "PRODUCT"]]
    return labels

# 1. DATA LOADING AND PREPROCESSING
def load_data(file_path):
    """Load the pre-annotated dataset"""
    df = pd.read_csv(file_path)
    print(f"Loaded pre-annotated dataset with {len(df)} records")

    # Check for platform column and handle missing values
    if 'platform' not in df.columns or df['platform'].isnull().all():
        print("Warning: No platform information found. Creating generic platform value.")
        df['platform'] = 'generic'
    else:
        # Fill missing platform values
        df['platform'] = df['platform'].fillna('unknown')

    # Display platform distribution
    platform_counts = df['platform'].value_counts()
    print("\nPlatform distribution:")
    for platform, count in platform_counts.items():
        print(f"  {platform}: {count} ({count / len(df) * 100:.2f}%)")

    return df


def preprocess_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ''

    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into text
    return ' '.join(tokens)


def prepare_data(df):
    """Prepare dataset and perform full advanced preprocessing (no opinion filter)"""
    # Combine title and text for submissions
    df['content'] = df.apply(
        lambda row: f"{row['title']} {row['text']}" if row['type'] == 'submission' else row['text'],
        axis=1
    )

    # Drop rows with missing data
    df = df.dropna(subset=['content', 'manual_sentiment'])

    # Step 1: clean raw text
    print("Cleaning and tokenizing...")
    df['processed_content'] = df['content'].apply(preprocess_text)

    # Step 2: apply WSD — store separately
    print("Applying WSD...")
    df['wsd_sense_content'] = df['processed_content'].apply(apply_wsd)

    # Step 3: sarcasm detection — use processed (not WSD) input
    print("Detecting sarcasm...")
    sarcasm_preds, sarcasm_confs = detect_sarcasm_batch(df['content'].tolist())
    df['is_sarcastic'] = sarcasm_preds
    df['sarcasm_confidence'] = sarcasm_confs

    # Step 4: named entity recognition
    print("Extracting named entities...")
    ner_labels = df['content'].apply(lambda x: extract_named_entity_labels(str(x)))
    for label in ["PERSON", "ORG", "PRODUCT"]:
        df[f'has_entity_{label.lower()}'] = ner_labels.apply(lambda ents: 1 if label in ents else 0)

    # Step 5: convert sentiment to binary
    df['sentiment_binary'] = df['manual_sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    # Step 6: aspect annotations (if any)
    for col in ['manual_content_quality', 'manual_pricing', 'manual_ui_ux',
                'manual_technical', 'manual_customer_service']:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[f"{col}_binary"] = df[col].apply(lambda x: 1 if x > 0 else 0)
            else:
                df[f"{col}_binary"] = df[col]

    return df

# 2. PLATFORM-SPECIFIC FEATURE ENGINEERING
def extract_platform_features(df):
    """Extract platform-specific features"""
    # Get unique platforms
    platforms = df['platform'].unique().tolist()

    print(f"Detected platforms: {', '.join(platforms)}")

    # Create platform indicator features
    for platform in platforms:
        platform_col_name = f'platform_{platform.lower().replace(" ", "_")}'
        df[platform_col_name] = (df['platform'] == platform).astype(int)

    # Platform-specific keyword dictionaries
    platform_keywords = {
        'netflix': ['netflix', 'nflx', 'netflix original', 'reed hastings', 'stranger things', 'squid game'],
        'apple_tv': ['apple tv', 'apple tv+', 'apple original', 'apple+', 'ted lasso', 'morning show'],
        'disney': ['disney', 'disney+', 'disney plus', 'mandalorian', 'marvel', 'pixar'],
        'hulu': ['hulu'],
        'amazon': ['prime video', 'amazon prime', 'prime', 'jeff bezos'],
        'hbo': ['hbo', 'hbo max', 'max', 'house of the dragon', 'game of thrones'],
        'peacock': ['peacock', 'nbc'],
        'paramount': ['paramount', 'paramount+', 'cbs'],
        'general': ['streaming', 'subscription', 'platform']
    }

    # Add platform-specific keyword features for platforms that actually exist in our data
    for platform_key, keywords in platform_keywords.items():
        # Check if we have this platform or something similar
        platform_exists = any(p.lower().replace(" ", "_") in platform_key or
                              platform_key in p.lower().replace(" ", "_")
                              for p in platforms)

        if platform_exists or platform_key == 'general':
            keyword_pattern = '|'.join([re.escape(kw) for kw in keywords])
            df[f'mentions_{platform_key}'] = df['content'].apply(
                lambda x: 1 if isinstance(x, str) and re.search(keyword_pattern, x.lower()) else 0
            )

    # Platform competition mentions (when a comment mentions multiple platforms)
    mention_cols = [col for col in df.columns if col.startswith('mentions_') and col != 'mentions_general']
    if mention_cols:
        df['mentions_multiple_platforms'] = (df[mention_cols].sum(axis=1) > 1).astype(int)

    # Print the platform-related columns we created
    platform_cols = [col for col in df.columns if col.startswith('platform_')]
    mention_cols = [col for col in df.columns if col.startswith('mentions_')]
    print(f"Created {len(platform_cols)} platform indicator columns: {', '.join(platform_cols)}")
    print(f"Created {len(mention_cols)} platform mention columns: {', '.join(mention_cols)}")

    return df


# 3. ENHANCED FEATURE ENGINEERING
def extract_features(df):
    """Extract additional features from the text"""
    # Length features
    df['text_length'] = df['content'].apply(lambda x: len(str(x)))
    df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))

    # General streaming service keywords
    streaming_keywords = ['stream', 'subscription', 'series', 'show', 'movie', 'watch', 'binge', 'content', 'original']
    for keyword in streaming_keywords:
        df[f'has_{keyword}'] = df['content'].apply(lambda x: 1 if keyword in str(x).lower() else 0)

    # Define aspect-related keywords
    aspect_keywords = define_cross_platform_aspects()

    for aspect, keywords in aspect_keywords.items():
        df[f'kw_{aspect}'] = df['content'].apply(
            lambda x: sum(1 for kw in keywords if kw in str(x).lower())
        )

    # Sentiment lexicon features
    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'like', 'best', 'awesome', 'worth', 'recommend']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'worst', 'poor', 'waste', 'expensive', 'cancel']

    df['positive_word_count'] = df['content'].apply(
        lambda x: sum(1 for word in positive_words if word in str(x).lower())
    )
    df['negative_word_count'] = df['content'].apply(
        lambda x: sum(1 for word in negative_words if word in str(x).lower())
    )
    df['sentiment_ratio'] = df.apply(
        lambda row: row['positive_word_count'] / (row['negative_word_count'] + 1), axis=1
    )

    # Platform-specific features
    df = extract_platform_features(df)

    return df


# Define cross-platform aspects
def define_cross_platform_aspects():
    """Define aspect keywords that work across streaming platforms"""
    return {
        'content_quality': [
            'quality', 'content', 'show', 'movie', 'series', 'documentary',
            'original', 'programming', 'catalog', 'library', 'selection'
        ],
        'pricing': [
            'price', 'cost', 'subscription', 'fee', 'expensive', 'cheap',
            'worth', 'pay', 'money', 'value', 'plan', 'tier'
        ],
        'ui_ux': [
            'interface', 'design', 'ui', 'ux', 'navigation', 'search',
            'find', 'browse', 'menu', 'layout', 'usability', 'app'
        ],
        'technical': [
            'buffer', 'stream', 'load', 'quality', 'hd', '4k', 'resolution',
            'error', 'bug', 'crash', 'playback', 'bandwidth', 'offline'
        ],
        'customer_service': [
            'support', 'service', 'help', 'contact', 'response', 'customer',
            'chat', 'email', 'refund', 'cancel', 'subscription'
        ]
    }

def main():
    # Set the path to your pre-annotated dataset
    file_path = '../data/evaluation_dataset_merged.csv'

    # Configure warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Step 1: Load data
    print("Loading dataset...")
    df = load_data(file_path)

    # Step 2: Prepare data (includes cleaning, WSD, sarcasm detection, NER, sentiment, aspects)
    print("Preparing data...")
    df = prepare_data(df)

    # Step 3: Feature extraction
    print("\nExtracting additional features...")
    df = extract_features(df)

    # Step 4: Display summary
    print(f"\nProcessed {len(df)} records")
    print(f"Positive sentiment: {df['sentiment_binary'].sum()} ({df['sentiment_binary'].mean() * 100:.2f}%)")
    print(f"Negative sentiment: {len(df) - df['sentiment_binary'].sum()}")

    # Step 5: Save output
    print("Saving enhanced dataset...")
    os.makedirs("result", exist_ok=True)
    df.to_csv("result/streaming_enhanced_features.csv", index=False)
    print("Saved to result/streaming_enhanced_features.csv")


if __name__ == "__main__":
    main()
