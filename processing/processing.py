import pandas as pd
import numpy as np
import re
import nltk
import json
import hashlib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)


class EnhancedDataProcessor:
    """
    Enhanced data processor for streaming opinions with comprehensive text analysis capabilities
    including sentiment analysis, feature extraction, entity recognition, and language detection.
    """

    def __init__(self,
                 streaming_platforms=None,
                 feature_lexicons=None,
                 min_text_length=5,
                 duplicate_threshold=0.8):
        """
        Initialize the enhanced data processor.

        Args:
            streaming_platforms (dict): Dictionary of streaming platforms and their keywords
            feature_lexicons (dict): Dictionary of feature categories and their keywords
            min_text_length (int): Minimum word count for text to be processed
            duplicate_threshold (float): Threshold for considering documents as duplicates
        """
        # Core NLP components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Set processing parameters
        self.min_text_length = min_text_length
        self.duplicate_threshold = duplicate_threshold

        # Default streaming platforms if none provided
        self.streaming_platforms = streaming_platforms or {
            'netflix': ['netflix', 'netflix\'s'],
            'disney+': ['disney+', 'disney plus', 'disneyplus'],
            'hbo max': ['hbo max', 'hbomax', 'hbo'],
            'amazon prime': ['amazon prime', 'prime video', 'primevideo'],
            'hulu': ['hulu', 'hulu\'s'],
            'apple tv+': ['apple tv+', 'apple tv plus', 'appletv+'],
            'peacock': ['peacock'],
            'paramount+': ['paramount+', 'paramount plus']
        }

        # Default feature lexicons if none provided
        self.feature_lexicons = feature_lexicons or {
            'content_quality': [
                'content', 'show', 'movie', 'series', 'documentary', 'original', 'library',
                'catalog', 'selection', 'variety', 'quality', 'production', 'story', 'plot',
                'acting', 'writing', 'entertainment', 'binge', 'watch'
            ],
            'pricing': [
                'price', 'cost', 'subscription', 'fee', 'month', 'annual', 'plan', 'tier',
                'premium', 'basic', 'standard', 'free', 'trial', 'worth', 'value', 'money',
                'expensive', 'cheap', 'affordable', 'discount', 'deal', 'bundle'
            ],
            'ui_ux': [
                'interface', 'design', 'app', 'application', 'website', 'navigation', 'browse',
                'search', 'find', 'recommendation', 'suggest', 'algorithm', 'profile', 'user',
                'experience', 'ui', 'ux', 'layout', 'menu', 'feature', 'usability', 'friendly'
            ],
            'technical': [
                'quality', 'resolution', 'hd', '4k', 'uhd', 'audio', 'video', 'stream', 'buffer',
                'lag', 'loading', 'download', 'offline', 'speed', 'performance', 'fast', 'slow',
                'playback', 'device', 'compatibility', 'support', 'bug', 'crash', 'freeze'
            ],
            'customer_service': [
                'support', 'help', 'service', 'customer', 'representative', 'contact', 'response',
                'issue', 'problem', 'solve', 'resolution', 'refund', 'cancel', 'subscription',
                'chat', 'email', 'phone', 'wait', 'time', 'responsive', 'helpful'
            ]
        }

        self.bot_list = ['RemindMeBot', 'AutoModerator']

    def load_data(self, filepath):
        """
        Load data from CSV file.

        Args:
            filepath (str): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        return df

    def clean_text(self, text):
        """
        Clean and normalize text.

        Args:
            text (str): Text to clean

        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Replacing multi spaces/new lines to one space
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]

        # Join tokens back to text
        cleaned_text = ' '.join(tokens)
        return cleaned_text

    def detect_language(self, text):
        """
        Detect the language of the text.

        Args:
            text (str): Text to detect language for

        Returns:
            str: ISO language code or 'unknown'
        """
        if not text or len(text) < 20:
            return 'unknown'

        try:
            return detect(text)
        except:
            return 'unknown'

    def detect_streaming_platform(self, text):
        """
        Detect which streaming platform is being discussed in the text.

        Args:
            text (str): Text to analyze

        Returns:
            str: Detected platform name or 'general' if none found
        """
        if not text:
            return 'general'

        text = text.lower()
        for platform, keywords in self.streaming_platforms.items():
            for keyword in keywords:
                if keyword in text:
                    return platform
        return 'general'

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of the text.

        Args:
            text (str): Text to analyze

        Returns:
            tuple: (sentiment category, sentiment score)
        """
        if not text:
            return 'neutral', 0.0

        scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = scores['compound']

        # Convert score to sentiment category
        if compound_score >= 0.05:
            sentiment = 'positive'
        elif compound_score <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return sentiment, compound_score

    def extract_features(self, text):
        """
        Extract scores for different features (content quality, pricing, etc.).

        Args:
            text (str): Text to analyze

        Returns:
            dict: Feature scores
        """
        if not text:
            return {feature: 0.0 for feature in self.feature_lexicons}

        text = text.lower()
        feature_scores = {}

        # Calculate sentiment for each feature based on relevant keywords
        for feature, keywords in self.feature_lexicons.items():
            feature_text = ""
            for keyword in keywords:
                # Extract sentences containing feature keywords
                pattern = r'\b' + re.escape(keyword) + r'\b(?:\s+\w+)*'
                matches = re.findall(pattern, text)
                if matches:
                    feature_text += ' '.join(matches) + ' '

            # Calculate sentiment for the feature text
            if feature_text:
                _, sentiment_score = self.analyze_sentiment(feature_text)
                feature_scores[feature] = sentiment_score
            else:
                feature_scores[feature] = 0.0

        return feature_scores

    def extract_entities(self, text):
        """
        Extract named entities from text.

        Args:
            text (str): Text to analyze

        Returns:
            list: List of entities
        """
        if not text or len(text) < 20:
            return []

        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)

            entities = []
            current_entity = []

            for subtree in named_entities:
                if isinstance(subtree, nltk.Tree):
                    entity_type = subtree.label()
                    entity_text = ' '.join([word for word, tag in subtree.leaves()])
                    entities.append({
                        'text': entity_text,
                        'type': entity_type
                    })

            return entities
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []

    def extract_keywords(self, text, top_n=10):
        """
        Extract important keywords from text using TF-IDF.

        Args:
            text (str): Text to analyze
            top_n (int): Number of top keywords to extract

        Returns:
            list: List of keywords
        """
        if not text or len(text) < 20:
            return []

        try:
            # Use CountVectorizer for single document keyword extraction
            vectorizer = CountVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                max_features=100
            )

            # Fit and transform the text
            X = vectorizer.fit_transform([text])

            # Get feature names and their counts
            feature_names = vectorizer.get_feature_names_out()
            feature_counts = X.toarray()[0]

            # Sort by count and get top keywords
            keywords = [
                           feature_names[i] for i in feature_counts.argsort()[::-1]
                           if feature_counts[i] > 0
                       ][:top_n]

            return keywords
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []

    def preprocess_data(self, df):
        """
        Comprehensive preprocessing of the dataset.

        Args:
            df (pd.DataFrame): DataFrame to preprocess

        Returns:
            pd.DataFrame: Processed DataFrame
        """
        print("Starting comprehensive preprocessing...")

        # Create a unique ID column if not exists
        if 'id' not in df.columns:
            df['id'] = [f"record_{i}" for i in range(len(df))]

        # Remove bot comments
        if 'author' in df.columns:
            print("Removing bot comments...")
            bot_comments = df[df['author'].isin(self.bot_list)].shape[0]
            df = df[~df['author'].isin(self.bot_list)]
            print(f'Removed {bot_comments} bot comments')

        # Clean text columns
        if 'text' in df.columns:
            print("Cleaning text...")
            df['cleaned_text'] = df['text'].apply(self.clean_text)

        # Combine title and text if both exist
        if 'title' in df.columns and 'text' in df.columns:
            print("Combining title and text...")
            df['full_text'] = df.apply(
                lambda row: (str(row['title']) + " " + str(row['text']))
                if pd.notna(row['title']) else str(row['text']),
                axis=1
            )
            df['cleaned_full_text'] = df['full_text'].apply(self.clean_text)

        # Use cleaned_full_text if available, otherwise use cleaned_text
        text_column = 'cleaned_full_text' if 'cleaned_full_text' in df.columns else 'cleaned_text'

        # Remove very short texts
        print("Filtering out short texts...")
        df = df[df[text_column].str.split().str.len() > self.min_text_length]

        # Detect language
        print("Detecting language...")
        df['language'] = df[text_column].apply(self.detect_language)

        # Filter for English texts
        df = df[df['language'] == 'en']

        # Detect platform if not already present
        if 'platform' not in df.columns:
            print("Detecting streaming platforms...")
            df['platform'] = df[text_column].apply(self.detect_streaming_platform)

        # Analyze sentiment
        print("Analyzing sentiment...")
        sentiment_results = df[text_column].apply(self.analyze_sentiment)
        df['sentiment'] = sentiment_results.apply(lambda x: x[0])
        df['sentiment_score'] = sentiment_results.apply(lambda x: x[1])

        # Extract feature scores
        print("Extracting feature scores...")
        feature_scores = df[text_column].apply(self.extract_features)

        # Add feature scores as separate columns
        for feature in self.feature_lexicons.keys():
            df[feature] = feature_scores.apply(lambda x: x.get(feature, 0.0))

        # Extract entities and keywords
        print("Extracting entities and keywords...")
        df['entities'] = df[text_column].apply(lambda x: json.dumps(self.extract_entities(x)))
        df['keywords'] = df[text_column].apply(lambda x: json.dumps(self.extract_keywords(x)))

        print(f"Preprocessing complete. {len(df)} records remain.")
        return df

    def detect_duplicates(self, df, text_column='cleaned_text', threshold=None):
        """
        Detect near-duplicate content using TF-IDF and cosine similarity.

        Args:
            df (pd.DataFrame): DataFrame to check for duplicates
            text_column (str): Column containing cleaned text
            threshold (float): Similarity threshold for considering documents as duplicates

        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        threshold = threshold or self.duplicate_threshold
        print(f"Detecting near-duplicate content with threshold {threshold}...")

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
        """
        Balance the dataset by sentiment.

        Args:
            df (pd.DataFrame): DataFrame to balance
            sentiment_column (str): Column containing sentiment labels
            target_ratio (float): Target ratio for positive:negative samples

        Returns:
            pd.DataFrame: Balanced DataFrame
        """
        sentiment_counts = df[sentiment_column].value_counts()
        print(f"Initial sentiment distribution: {sentiment_counts}")

        # Filter out neutral sentiment if present
        if 'neutral' in sentiment_counts:
            print("Removing neutral sentiment for balancing...")
            df_filtered = df[df[sentiment_column] != 'neutral']
            sentiment_counts = df_filtered[sentiment_column].value_counts()
        else:
            df_filtered = df

        # If already balanced or empty, return as is
        if len(sentiment_counts) < 2 or abs(
                sentiment_counts.get('positive', 0) / len(df_filtered) - target_ratio) < 0.05:
            print("Dataset already balanced or too small to balance.")
            return df

        # Determine majority and minority classes
        majority_sentiment = sentiment_counts.idxmax()
        minority_sentiment = sentiment_counts.idxmin()

        # Calculate how many records to keep from majority class
        target_majority_count = int(sentiment_counts[minority_sentiment] / target_ratio * (1 - target_ratio))

        # Sample majority class
        majority_df = df_filtered[df_filtered[sentiment_column] == majority_sentiment].sample(
            n=min(target_majority_count, sentiment_counts[majority_sentiment]),
            random_state=42
        )

        # Combine with minority class
        minority_df = df_filtered[df_filtered[sentiment_column] == minority_sentiment]

        # Add back neutral sentiment if it was present
        if 'neutral' in df[sentiment_column].unique():
            neutral_df = df[df[sentiment_column] == 'neutral']
            balanced_df = pd.concat([majority_df, minority_df, neutral_df])
        else:
            balanced_df = pd.concat([majority_df, minority_df])

        print(f"Balanced sentiment distribution: {balanced_df[sentiment_column].value_counts()}")
        return balanced_df

    def generate_corpus_statistics(self, df, text_column='cleaned_text'):
        """
        Generate comprehensive statistics about the corpus.

        Args:
            df (pd.DataFrame): DataFrame to analyze
            text_column (str): Column containing cleaned text

        Returns:
            dict: Corpus statistics
        """
        print("Generating corpus statistics...")

        # Count total number of records
        num_records = len(df)

        # Count total number of words
        df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
        total_words = df['word_count'].sum()

        # Count unique words (types)
        all_words = ' '.join(df[text_column].fillna('').astype(str)).split()
        unique_words = set(all_words)
        num_types = len(unique_words)

        # Get distribution statistics
        distributions = {}

        # Get categorical distributions
        for column in ['platform', 'source', 'sentiment', 'language']:
            if column in df.columns:
                distributions[f"{column}_distribution"] = df[column].value_counts().to_dict()

        # Get feature score averages
        feature_stats = {}
        for feature in self.feature_lexicons.keys():
            if feature in df.columns:
                feature_stats[feature] = {
                    'mean': df[feature].mean(),
                    'median': df[feature].median(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max()
                }

        # Get text length statistics
        text_length_stats = {
            'mean_word_count': df['word_count'].mean(),
            'median_word_count': df['word_count'].median(),
            'min_word_count': df['word_count'].min(),
            'max_word_count': df['word_count'].max()
        }

        # Compile all statistics
        statistics = {
            'num_records': num_records,
            'total_words': total_words,
            'num_unique_words': num_types,
            'lexical_diversity': num_types / total_words if total_words > 0 else 0,
            'distributions': distributions,
            'feature_stats': feature_stats,
            'text_length_stats': text_length_stats
        }

        return statistics

    def save_processed_data(self, df, output_filepath):
        """
        Save processed data to CSV.

        Args:
            df (pd.DataFrame): Processed DataFrame
            output_filepath (str): Path to save the CSV file
        """
        df.to_csv(output_filepath, index=False)
        print(f"Saved processed data with {len(df)} records to {output_filepath}")

    def process_pipeline(self, input_filepath, output_filepath, balance_data=True):
        """
        Run the complete processing pipeline.

        Args:
            input_filepath (str): Path to input CSV file
            output_filepath (str): Path to save processed CSV file
            balance_data (bool): Whether to balance the dataset by sentiment

        Returns:
            tuple: (processed_df, statistics)
        """
        print(f"Starting processing pipeline for {input_filepath}")

        # Load raw data
        raw_df = self.load_data(input_filepath)

        # Preprocess data (includes cleaning, sentiment analysis, etc.)
        processed_df = self.preprocess_data(raw_df)

        # Remove duplicates
        deduplicated_df = self.detect_duplicates(processed_df)

        # Balance sentiment if requested
        if balance_data:
            balanced_df = self.balance_sentiment(deduplicated_df)
        else:
            balanced_df = deduplicated_df

        # Generate corpus statistics
        statistics = self.generate_corpus_statistics(balanced_df)

        # Save processed data
        self.save_processed_data(balanced_df, output_filepath)

        print(f"Processing pipeline completed. Output saved to {output_filepath}")

        return balanced_df, statistics


# Example usage
if __name__ == "__main__":
    processor = EnhancedDataProcessor()

    # Run the complete pipeline
    processed_df, statistics = processor.process_pipeline(
        input_filepath="../data/streaming_opinions_dataset.csv",
        output_filepath="../data/processed_streaming_opinions.csv",
        balance_data=True
    )

    # Print corpus statistics
    print("\nCorpus Statistics:")
    for key, value in statistics.items():
        if not isinstance(value, dict):
            print(f"- {key}: {value}")
        else:
            print(f"- {key}:")
            for sub_key, sub_value in value.items():
                print(f"  - {sub_key}: {sub_value}")