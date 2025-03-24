import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from tqdm import tqdm

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class DataProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def load_data(self, filepath):
        """Load data from CSV file"""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        return df

    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]

        # Join tokens back to text
        cleaned_text = ' '.join(tokens)
        return cleaned_text

    def preprocess_data(self, df):
        """Preprocess the dataset"""
        # Create a unique ID column if not exists
        if 'id' not in df.columns:
            df['id'] = [f"record_{i}" for i in range(len(df))]

        # Clean text columns
        if 'text' in df.columns:
            df['cleaned_text'] = df['text'].apply(self.clean_text)

        if 'title' in df.columns and 'text' in df.columns:
            # Combine title and text for analysis
            df['full_text'] = df.apply(lambda row:
                                       (str(row['title']) + " " + str(row['text']))
                                       if pd.notna(row['title']) else str(row['text']),
                                       axis=1)
            df['cleaned_full_text'] = df['full_text'].apply(self.clean_text)

        # Remove very short texts (likely not useful for analysis)
        df = df[df['cleaned_text'].str.split().str.len() > 5]

        return df

    def detect_duplicates(self, df, text_column='cleaned_text', threshold=0.8):
        """Detect near-duplicate content using TF-IDF and cosine similarity"""
        print("Detecting near-duplicate content...")

        # Get non-empty texts
        texts = df[text_column].dropna().tolist()
        if len(texts) < 2:
            return df

        # Create TF-IDF matrix
        tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        # Calculate pairwise similarity
        duplicate_indices = set()

        # For large datasets, process in batches to avoid memory issues
        batch_size = 1000
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_end = min(i + batch_size, len(texts))
            batch_matrix = tfidf_matrix[i:batch_end]

            # Calculate cosine similarity between this batch and all documents
            similarities = cosine_similarity(batch_matrix, tfidf_matrix)

            # Find duplicates
            for batch_idx, sim_scores in enumerate(similarities):
                doc_idx = i + batch_idx
                # Find similar documents (excluding self-comparison)
                similar_indices = np.where(sim_scores > threshold)[0]

                for similar_idx in similar_indices:
                    if similar_idx != doc_idx and similar_idx > doc_idx:
                        # Keep the document with more content or higher engagement
                        if len(texts[doc_idx]) < len(texts[similar_idx]):
                            duplicate_indices.add(doc_idx)
                        else:
                            duplicate_indices.add(similar_idx)

        # Create a duplicate flag
        df['is_duplicate'] = df.index.isin(duplicate_indices)

        # Filter out duplicates
        df_no_duplicates = df[~df['is_duplicate']]

        print(f"Removed {len(duplicate_indices)} duplicate records. {len(df_no_duplicates)} records remaining.")
        return df_no_duplicates

    def balance_sentiment(self, df, sentiment_column='sentiment', target_ratio=0.5):
        """Balance the dataset by sentiment"""
        sentiment_counts = df[sentiment_column].value_counts()
        print(f"Initial sentiment distribution: {sentiment_counts}")

        # If already balanced, return as is
        if abs(sentiment_counts.get('positive', 0) / len(df) - target_ratio) < 0.05:
            return df

        # Determine majority and minority classes
        majority_sentiment = sentiment_counts.idxmax()
        minority_sentiment = sentiment_counts.idxmin()

        # Calculate how many records to keep from majority class
        target_majority_count = int(sentiment_counts[minority_sentiment] / target_ratio * (1 - target_ratio))

        # Sample majority class
        majority_df = df[df[sentiment_column] == majority_sentiment].sample(
            n=min(target_majority_count, sentiment_counts[majority_sentiment]),
            random_state=42
        )

        # Combine with minority class
        minority_df = df[df[sentiment_column] == minority_sentiment]
        balanced_df = pd.concat([majority_df, minority_df])

        print(f"Balanced sentiment distribution: {balanced_df[sentiment_column].value_counts()}")
        return balanced_df

    def save_processed_data(self, df, output_filepath):
        """Save processed data to CSV"""
        df.to_csv(output_filepath, index=False)
        print(f"Saved processed data with {len(df)} records to {output_filepath}")

    def generate_corpus_statistics(self, df, text_column='cleaned_text'):
        """Generate statistics about the corpus"""
        # Count total number of records
        num_records = len(df)

        # Count total number of words
        df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
        total_words = df['word_count'].sum()

        # Count unique words (types)
        all_words = ' '.join(df[text_column].fillna('').astype(str)).split()
        unique_words = set(all_words)
        num_types = len(unique_words)

        # Get distribution by platform
        platform_distribution = df['platform'].value_counts() if 'platform' in df.columns else None

        # Get distribution by source
        source_distribution = df['source'].value_counts() if 'source' in df.columns else None

        statistics = {
            'num_records': num_records,
            'total_words': total_words,
            'num_types': num_types,
            'avg_words_per_record': total_words / num_records,
            'platform_distribution': platform_distribution,
            'source_distribution': source_distribution
        }

        return statistics


# Example usage
if __name__ == "__main__":
    processor = DataProcessor()

    # Load raw data
    raw_df = processor.load_data("../data/streaming_opinions_dataset.csv")

    # Preprocess data
    processed_df = processor.preprocess_data(raw_df)

    # Remove duplicates
    deduplicated_df = processor.detect_duplicates(processed_df)

    # Generate corpus statistics
    statistics = processor.generate_corpus_statistics(deduplicated_df)
    print("Corpus Statistics:")
    for key, value in statistics.items():
        if not isinstance(value, pd.Series):
            print(f"- {key}: {value}")
        else:
            print(f"- {key}:")
            print(value)

    # Save processed data
    processor.save_processed_data(deduplicated_df, "../data/processed_streaming_opinions.csv")