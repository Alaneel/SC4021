# auto_annotation.py
"""
Automated annotation system for sentiment classification evaluation datasets.
Replaces manual annotation with automated approaches.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
from tqdm import tqdm
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')


class AutoAnnotator:
    """System for automatically annotating evaluation datasets."""

    def __init__(self, config=None):
        """
        Initialize the auto annotator.

        Args:
            config (dict): Configuration settings for annotation methods
        """
        # Default configuration
        self.config = {
            'primary_method': 'ensemble',  # options: 'lexicon', 'ml', 'ensemble'
            'lexicon_weight': 0.5,
            'ml_weight': 0.5,
            'feature_extraction': True,
            'annotator_simulation': {
                'enabled': True,
                'count': 3,
                'agreement_level': 0.7  # 0-1 scale where 1 is perfect agreement
            },
            'random_seed': 42
        }

        # Override defaults with user config
        if config:
            self.config.update(config)

        # Initialize random seed
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])

        # Initialize sentiment analyzers
        self.lexicon_analyzer = SentimentIntensityAnalyzer()
        self.ml_classifier = None
        self.feature_extractors = {}

    def load_data(self, data_path):
        """
        Load dataset for annotation.

        Args:
            data_path (str): Path to the dataset

        Returns:
            pd.DataFrame: Loaded dataset
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        data = pd.read_csv(data_path)
        print(f"Loaded {len(data)} records from {data_path}")
        return data

    def train_ml_models(self, training_data, text_column='text', sentiment_column='sentiment', feature_columns=None):
        """
        Train machine learning models on existing data.

        Args:
            training_data (pd.DataFrame): Data with existing labels
            text_column (str): Column containing text content
            sentiment_column (str): Column containing sentiment labels
            feature_columns (list): Columns containing feature scores

        Returns:
            dict: Trained models
        """
        print("Training ML models...")

        # Prepare text data
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.8,
            stop_words='english'
        )

        # Handle potentially missing text
        training_data[text_column] = training_data[text_column].fillna("")

        X_text = vectorizer.fit_transform(training_data[text_column])
        y_sentiment = training_data[sentiment_column]

        # Train sentiment classifier
        sentiment_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config['random_seed']
        )
        sentiment_classifier.fit(X_text, y_sentiment)

        self.ml_classifier = {
            'vectorizer': vectorizer,
            'classifier': sentiment_classifier
        }

        # Train feature extractors if enabled
        if self.config['feature_extraction'] and feature_columns:
            for feature in feature_columns:
                if feature in training_data.columns:
                    # Only train on rows where feature value exists
                    valid_indices = ~training_data[feature].isna()

                    if valid_indices.sum() > 100:  # Ensure we have enough data
                        X_feature = X_text[valid_indices]
                        y_feature = training_data.loc[valid_indices, feature]

                        # Train regression model for this feature
                        model = Ridge(alpha=1.0, random_state=self.config['random_seed'])
                        model.fit(X_feature, y_feature)

                        self.feature_extractors[feature] = {
                            'vectorizer': vectorizer,  # Reuse the same vectorizer
                            'model': model
                        }

                        print(f"Trained feature extractor for {feature}")

        return {
            'sentiment': self.ml_classifier,
            'features': self.feature_extractors
        }

    def lexicon_based_annotation(self, text):
        """
        Perform lexicon-based sentiment analysis.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Sentiment scores and classification
        """
        if not text or pd.isna(text):
            return {
                'sentiment': 'neutral',
                'scores': {
                    'pos': 0.0,
                    'neg': 0.0,
                    'neu': 1.0,
                    'compound': 0.0
                }
            }

        # Get VADER sentiment scores
        scores = self.lexicon_analyzer.polarity_scores(text)

        # Determine sentiment class
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'sentiment': sentiment,
            'scores': scores
        }

    def ml_based_annotation(self, text):
        """
        Perform ML-based sentiment analysis.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Sentiment classification and probabilities
        """
        if not self.ml_classifier:
            raise ValueError("ML classifier has not been trained")

        if not text or pd.isna(text):
            return {
                'sentiment': 'neutral',
                'probabilities': {
                    'positive': 0.33,
                    'negative': 0.33,
                    'neutral': 0.34
                }
            }

        # Vectorize text
        vectorizer = self.ml_classifier['vectorizer']
        classifier = self.ml_classifier['classifier']

        X = vectorizer.transform([text])

        # Predict sentiment
        sentiment = classifier.predict(X)[0]
        probabilities = classifier.predict_proba(X)[0]

        # Map probabilities to sentiment classes
        classes = classifier.classes_
        prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}

        return {
            'sentiment': sentiment,
            'probabilities': prob_dict
        }

    def extract_features(self, text):
        """
        Extract feature scores from text.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Extracted feature scores
        """
        features = {}

        # Skip if text is empty
        if not text or pd.isna(text):
            return {feature: 0.0 for feature in self.feature_extractors}

        # Rule-based feature extraction
        # These are simplified heuristics and should be enhanced for production use

        # Content quality: Based on length, capitalization, punctuation
        words = word_tokenize(text) if text else []
        word_count = len(words)
        avg_word_length = np.mean([len(w) for w in words]) if words else 0

        # Simple heuristic for content quality
        quality_score = min(1.0, max(-1.0,
                                     (word_count / 100 - 0.5) + (avg_word_length / 10 - 0.5)
                                     ))
        features['content_quality'] = quality_score

        # Pricing mentions
        price_terms = ['price', 'cost', 'expensive', 'cheap', 'affordable', 'pricing',
                       'subscription', 'fee', 'plan', 'pay', 'worth', '$', 'dollar', 'money']
        price_matches = sum(1 for term in price_terms if term.lower() in text.lower())

        vader_scores = self.lexicon_analyzer.polarity_scores(text)

        if price_matches > 0:
            price_score = vader_scores['compound'] * min(1.0, price_matches / 3)
        else:
            price_score = 0.0
        features['pricing'] = price_score

        # UI/UX mentions
        ui_terms = ['interface', 'design', 'ui', 'ux', 'usability', 'user experience',
                    'look and feel', 'layout', 'navigation', 'intuitive', 'clean']
        ui_matches = sum(1 for term in ui_terms if term.lower() in text.lower())

        if ui_matches > 0:
            ui_score = vader_scores['compound'] * min(1.0, ui_matches / 3)
        else:
            ui_score = 0.0
        features['ui_ux'] = ui_score

        # Technical mentions
        tech_terms = ['bug', 'crash', 'error', 'performance', 'loading', 'slow', 'fast',
                      'reliable', 'stable', 'feature', 'functionality']
        tech_matches = sum(1 for term in tech_terms if term.lower() in text.lower())

        if tech_matches > 0:
            tech_score = vader_scores['compound'] * min(1.0, tech_matches / 3)
        else:
            tech_score = 0.0
        features['technical'] = tech_score

        # Customer service mentions
        cs_terms = ['support', 'customer service', 'help', 'response', 'responsive',
                    'staff', 'team', 'assistance', 'representative', 'service']
        cs_matches = sum(1 for term in cs_terms if term.lower() in text.lower())

        if cs_matches > 0:
            cs_score = vader_scores['compound'] * min(1.0, cs_matches / 3)
        else:
            cs_score = 0.0
        features['customer_service'] = cs_score

        # If ML models are available, use them to refine feature scores
        if self.feature_extractors:
            text_vector = self.ml_classifier['vectorizer'].transform([text])

            for feature, extractor in self.feature_extractors.items():
                if feature in features:  # Only update features we've extracted
                    ml_score = extractor['model'].predict(text_vector)[0]

                    # Combine rule-based and ML scores (equal weight)
                    features[feature] = (features[feature] + ml_score) / 2

        # Ensure all scores are within [-1, 1] range
        for feature in features:
            features[feature] = max(-1.0, min(1.0, features[feature]))

        return features

    def annotate_record(self, record, text_column='text', title_column='title'):
        """
        Annotate a single record with sentiment and features.

        Args:
            record (Series): Record to annotate
            text_column (str): Column containing main text
            title_column (str): Column containing title text

        Returns:
            dict: Annotation results
        """
        # Combine title and text if both exist
        title = record.get(title_column, '')
        text = record.get(text_column, '')

        full_text = ''
        if not pd.isna(title) and title:
            full_text += title + ' '
        if not pd.isna(text) and text:
            full_text += text

        if not full_text:
            # No text to analyze
            return {
                'manual_sentiment': 'neutral',
                'features': {f'manual_{k}': 0.0 for k in
                             ['content_quality', 'pricing', 'ui_ux', 'technical', 'customer_service']}
            }

        # Perform sentiment analysis based on configured method
        if self.config['primary_method'] == 'lexicon':
            sentiment_result = self.lexicon_based_annotation(full_text)
            final_sentiment = sentiment_result['sentiment']
        elif self.config['primary_method'] == 'ml':
            if not self.ml_classifier:
                # Fall back to lexicon if ML not trained
                sentiment_result = self.lexicon_based_annotation(full_text)
                final_sentiment = sentiment_result['sentiment']
            else:
                sentiment_result = self.ml_based_annotation(full_text)
                final_sentiment = sentiment_result['sentiment']
        else:  # ensemble
            lexicon_result = self.lexicon_based_annotation(full_text)

            if self.ml_classifier:
                ml_result = self.ml_based_annotation(full_text)

                # Weighted ensemble
                lex_weight = self.config['lexicon_weight']
                ml_weight = self.config['ml_weight']

                # Get probability for each class
                sentiments = ['positive', 'negative', 'neutral']

                # Convert lexicon scores to pseudo-probabilities
                lex_compound = lexicon_result['scores']['compound']
                lex_probs = {
                    'positive': max(0, lex_compound),
                    'negative': max(0, -lex_compound),
                    'neutral': 1 - abs(lex_compound)
                }

                # Normalize to sum to 1
                lex_sum = sum(lex_probs.values())
                lex_probs = {k: v / lex_sum for k, v in lex_probs.items()}

                # Combine probabilities
                combined_probs = {}
                for sentiment in sentiments:
                    ml_prob = ml_result['probabilities'].get(sentiment, 0)
                    lex_prob = lex_probs.get(sentiment, 0)
                    combined_probs[sentiment] = (lex_prob * lex_weight) + (ml_prob * ml_weight)

                # Select sentiment with highest probability
                final_sentiment = max(combined_probs, key=combined_probs.get)
            else:
                final_sentiment = lexicon_result['sentiment']

        # Extract features
        features = self.extract_features(full_text)

        # Prepare output
        result = {
            'manual_sentiment': final_sentiment,
            'features': {f'manual_{k}': v for k, v in features.items()}
        }

        return result

    def add_annotator_variation(self, base_annotation, agreement_level=None):
        """
        Add variation to simulate different annotators.

        Args:
            base_annotation (dict): Base annotation
            agreement_level (float): Level of agreement (0-1)

        Returns:
            dict: Modified annotation
        """
        agreement_level = agreement_level or self.config['annotator_simulation']['agreement_level']

        # Make a copy of the base annotation
        varied_annotation = base_annotation.copy()

        # Randomly vary sentiment based on agreement level
        if random.random() > agreement_level:
            # Change sentiment to one of the other options
            current = base_annotation['manual_sentiment']
            options = [s for s in ['positive', 'negative', 'neutral'] if s != current]
            varied_annotation['manual_sentiment'] = random.choice(options)

        # Add noise to feature scores
        for feature, value in base_annotation['features'].items():
            # Calculate max variation based on agreement level
            max_variation = (1 - agreement_level) * 0.5

            # Add random noise
            noise = random.uniform(-max_variation, max_variation)
            new_value = value + noise

            # Ensure value stays within [-1, 1]
            varied_annotation['features'][feature] = max(-1.0, min(1.0, new_value))

        return varied_annotation

    def annotate_dataset(self, data, text_column='text', title_column='title',
                         output_path=None, progress_callback=None):
        """
        Annotate an entire dataset.

        Args:
            data (pd.DataFrame or str): DataFrame or path to CSV
            text_column (str): Column containing text
            title_column (str): Column containing title
            output_path (str): Path to save annotated data
            progress_callback (callable): Function to call with progress updates

        Returns:
            pd.DataFrame: Annotated dataset
        """
        # Load data if path provided
        if isinstance(data, str):
            data = self.load_data(data)

        if 'id' not in data.columns:
            data['id'] = data.index

        # Create copy for annotation
        annotated_df = data.copy()

        # Initialize columns for manual annotations
        annotated_df['manual_sentiment'] = None

        feature_columns = [
            'content_quality', 'pricing', 'ui_ux', 'technical', 'customer_service'
        ]

        for feature in feature_columns:
            annotated_df[f'manual_{feature}'] = None

        # Simulate multiple annotators if enabled
        simulate_annotators = self.config['annotator_simulation']['enabled']
        num_annotators = self.config['annotator_simulation']['count'] if simulate_annotators else 1

        # Create separate dataframes for each annotator
        annotator_dfs = {}
        for i in range(1, num_annotators + 1):
            annotator_dfs[f"annotator_{i}"] = annotated_df.copy()

        # Process each record
        print(f"Annotating {len(data)} records...")

        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Annotating"):
            # Get base annotation
            base_annotation = self.annotate_record(
                row, text_column=text_column, title_column=title_column
            )

            # For first annotator, use base annotation
            annotator_id = f"annotator_1"
            annotator_df = annotator_dfs[annotator_id]

            # Update sentiment
            annotator_df.at[idx, 'manual_sentiment'] = base_annotation['manual_sentiment']

            # Update feature scores
            for feature, value in base_annotation['features'].items():
                if feature in annotator_df.columns:
                    annotator_df.at[idx, feature] = value

            # Add timestamp and annotator ID
            annotator_df.at[idx, 'annotation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            annotator_df.at[idx, 'annotator_id'] = annotator_id

            # For additional annotators, add variation
            for i in range(2, num_annotators + 1):
                annotator_id = f"annotator_{i}"
                annotator_df = annotator_dfs[annotator_id]

                # Add variation to simulate different annotators
                varied_annotation = self.add_annotator_variation(base_annotation)

                # Update sentiment
                annotator_df.at[idx, 'manual_sentiment'] = varied_annotation['manual_sentiment']

                # Update feature scores
                for feature, value in varied_annotation['features'].items():
                    if feature in annotator_df.columns:
                        annotator_df.at[idx, feature] = value

                # Add timestamp and annotator ID
                annotator_df.at[idx, 'annotation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                annotator_df.at[idx, 'annotator_id'] = annotator_id

            # Call progress callback if provided
            if progress_callback and idx % 100 == 0:
                progress_callback(idx, len(data))

        # Save each annotator's results
        results = {}

        for annotator_id, annotator_df in annotator_dfs.items():
            if output_path:
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Generate annotator-specific filename
                base_name, ext = os.path.splitext(output_path)
                annotator_path = f"{base_name}_{annotator_id}{ext}"

                # Save to CSV
                annotator_df.to_csv(annotator_path, index=False)
                print(f"Saved {annotator_id} annotations to {annotator_path}")

                results[annotator_id] = {
                    'path': annotator_path,
                    'count': len(annotator_df)
                }
            else:
                results[annotator_id] = {
                    'count': len(annotator_df)
                }

        # Return first annotator's dataframe by default
        return annotator_dfs['annotator_1'], results

    def merge_annotations(self, annotator_dfs):
        """
        Merge annotations from multiple annotators.

        Args:
            annotator_dfs (dict): Dictionary of annotator DataFrames

        Returns:
            pd.DataFrame: Merged dataset
        """
        if len(annotator_dfs) == 1:
            # Only one annotator, just return that
            return list(annotator_dfs.values())[0]

        # Start with the first annotator's data
        first_annotator = list(annotator_dfs.keys())[0]
        merged_df = annotator_dfs[first_annotator].copy()

        # Get all record IDs
        all_ids = set()
        for df in annotator_dfs.values():
            all_ids.update(df['id'].values)

        # For each record, merge annotations
        for record_id in tqdm(all_ids, desc="Merging annotations"):
            # Collect sentiments for this record
            sentiments = {}
            feature_values = {}

            for annotator_id, df in annotator_dfs.items():
                record = df[df['id'] == record_id]

                if len(record) > 0:
                    # Count sentiment
                    sentiment = record['manual_sentiment'].iloc[0]
                    if sentiment:
                        sentiments[sentiment] = sentiments.get(sentiment, 0) + 1

                    # Collect feature values
                    for col in record.columns:
                        if col.startswith('manual_') and col != 'manual_sentiment':
                            if col not in feature_values:
                                feature_values[col] = []

                            value = record[col].iloc[0]
                            if not pd.isna(value):
                                feature_values[col].append(value)

            # Determine majority sentiment
            if sentiments:
                majority_sentiment = max(sentiments.items(), key=lambda x: x[1])[0]
                merged_df.loc[merged_df['id'] == record_id, 'manual_sentiment'] = majority_sentiment

            # Average feature values
            for feature, values in feature_values.items():
                if values:
                    avg_value = sum(values) / len(values)
                    merged_df.loc[merged_df['id'] == record_id, feature] = avg_value

        return merged_df


def automated_annotation_workflow(input_path, output_path, config=None, train_data_path=None):
    """
    Run the complete automated annotation workflow.

    Args:
        input_path (str): Path to input dataset
        output_path (str): Path to save annotated dataset
        config (dict): Configuration for annotation
        train_data_path (str): Path to training data with existing labels

    Returns:
        dict: Results of the annotation process
    """
    start_time = time.time()

    # Create auto annotator
    annotator = AutoAnnotator(config)

    # Train ML models if training data provided
    if train_data_path and os.path.exists(train_data_path):
        training_data = annotator.load_data(train_data_path)

        # Get feature columns
        feature_columns = [
            'content_quality', 'pricing', 'ui_ux', 'technical', 'customer_service'
        ]

        # Train models
        annotator.train_ml_models(
            training_data,
            text_column='text',
            sentiment_column='sentiment',
            feature_columns=feature_columns
        )

    # Load input data
    input_data = annotator.load_data(input_path)

    # Annotate dataset
    annotated_df, annotator_results = annotator.annotate_dataset(
        input_data,
        text_column='text',
        title_column='title',
        output_path=output_path
    )

    # Merge annotations if multiple annotators
    if len(annotator_results) > 1:
        # Load all annotator dataframes
        annotator_dfs = {}
        for annotator_id, result in annotator_results.items():
            if 'path' in result:
                annotator_dfs[annotator_id] = pd.read_csv(result['path'])

        # Merge annotations
        merged_df = annotator.merge_annotations(annotator_dfs)

        # Save merged dataset
        merged_path = output_path.replace('.csv', '_merged.csv')
        merged_df.to_csv(merged_path, index=False)

        print(f"Saved merged annotations to {merged_path}")

        annotator_results['merged'] = {
            'path': merged_path,
            'count': len(merged_df)
        }

    elapsed_time = time.time() - start_time

    print(f"Annotation complete in {elapsed_time:.2f} seconds")

    # Return results
    return {
        'input_records': len(input_data),
        'annotated_records': len(annotated_df),
        'annotators': annotator_results,
        'elapsed_time': elapsed_time
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Automated annotation for sentiment evaluation datasets')
    parser.add_argument('--input', '-i', required=True, help='Path to input dataset')
    parser.add_argument('--output', '-o', required=True, help='Path to save annotated dataset')
    parser.add_argument('--train', '-t', help='Path to training data with existing labels')
    parser.add_argument('--method', '-m', choices=['lexicon', 'ml', 'ensemble'], default='ensemble',
                        help='Primary annotation method')
    parser.add_argument('--annotators', '-a', type=int, default=3,
                        help='Number of simulated annotators')
    parser.add_argument('--agreement', type=float, default=0.7,
                        help='Agreement level between annotators (0-1)')

    args = parser.parse_args()

    # Configure annotator
    config = {
        'primary_method': args.method,
        'annotator_simulation': {
            'enabled': args.annotators > 1,
            'count': args.annotators,
            'agreement_level': args.agreement
        }
    }

    # Run workflow
    results = automated_annotation_workflow(
        args.input,
        args.output,
        config=config,
        train_data_path=args.train
    )

    print(json.dumps(results, indent=2))