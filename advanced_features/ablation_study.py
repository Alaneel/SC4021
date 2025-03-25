# evaluation/ablation_study.py
"""Ablation study for classification enhancements."""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Download required NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('wordnet')


class AblationStudy:
    """Class for performing ablation studies on classification enhancements."""

    def __init__(self, data_path=None, test_size=0.2, random_state=42):
        """
        Initialize the ablation study.

        Args:
            data_path (str): Path to the dataset
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        self.feature_importances = {}

        # Initialize enhancement modules
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def load_data(self, path=None):
        """
        Load the dataset.

        Args:
            path (str, optional): Path to the dataset

        Returns:
            pd.DataFrame: Loaded dataset
        """
        path = path or self.data_path
        if not path:
            raise ValueError("Data path must be provided")

        self.data = pd.read_csv(path)
        print(f"Loaded {len(self.data)} records from {path}")
        return self.data

    def prepare_data(self, text_column='text', label_column='sentiment'):
        """
        Prepare data for the ablation study.

        Args:
            text_column (str): Name of the text column
            label_column (str): Name of the label column

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            self.load_data()

        # Check that required columns exist
        if text_column not in self.data.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")

        if label_column not in self.data.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")

        # Remove rows with missing values
        self.data = self.data.dropna(subset=[text_column, label_column])

        # Split data
        X = self.data[text_column]
        y = self.data[label_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def clean_text(self, text):
        """
        Clean and normalize text.

        Args:
            text (str): Text to clean

        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        # Tokenize, remove stopwords, and lemmatize
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]

        # Join tokens back to text
        cleaned_text = ' '.join(tokens)
        return cleaned_text

    def extract_sentiment_features(self, text):
        """
        Extract sentiment features using VADER.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Sentiment scores
        """
        if not isinstance(text, str):
            return {'pos': 0, 'neg': 0, 'neu': 0, 'compound': 0}

        scores = self.sentiment_analyzer.polarity_scores(text)
        return scores

    def extract_text_stats(self, text):
        """
        Extract text statistics.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Text statistics
        """
        if not isinstance(text, str):
            return {'word_count': 0, 'avg_word_length': 0, 'char_count': 0}

        # Count words
        words = text.split()
        word_count = len(words)

        # Count characters
        char_count = len(text)

        # Calculate average word length
        avg_word_length = char_count / word_count if word_count > 0 else 0

        return {
            'word_count': word_count,
            'avg_word_length': avg_word_length,
            'char_count': char_count
        }

    def extract_platform_features(self, text):
        """
        Extract platform-specific features.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Platform features
        """
        if not isinstance(text, str):
            return {'has_netflix': 0, 'has_disney': 0, 'has_hulu': 0, 'has_amazon': 0, 'has_hbo': 0}

        text = text.lower()

        return {
            'has_netflix': 1 if 'netflix' in text else 0,
            'has_disney': 1 if 'disney' in text or 'disney+' in text else 0,
            'has_hulu': 1 if 'hulu' in text else 0,
            'has_amazon': 1 if 'amazon' in text or 'prime video' in text else 0,
            'has_hbo': 1 if 'hbo' in text or 'hbo max' in text else 0
        }

    def train_baseline_model(self, classifier='svm'):
        """
        Train a baseline model without any enhancements.

        Args:
            classifier (str): Classifier type ('svm', 'rf', or 'nb')

        Returns:
            dict: Model performance metrics
        """
        if self.X_train is None:
            self.prepare_data()

        # Create classifier
        if classifier == 'svm':
            clf = Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000)),
                ('classifier', SVC(probability=True))
            ])
        elif classifier == 'rf':
            clf = Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000)),
                ('classifier', RandomForestClassifier(n_estimators=100))
            ])
        elif classifier == 'nb':
            clf = Pipeline([
                ('vectorizer', CountVectorizer(max_features=5000)),
                ('classifier', MultinomialNB())
            ])
        else:
            raise ValueError(f"Unsupported classifier: {classifier}")

        # Train model
        clf.fit(self.X_train, self.y_train)

        # Evaluate model
        y_pred = clf.predict(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )

        # Store results
        results = {
            'model': clf,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'enhancements': []
        }

        self.results['baseline'] = results

        print(f"Baseline model ({classifier}):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return results

    def train_enhanced_model(self, enhancements=None, classifier='svm'):
        """
        Train a model with specified enhancements.

        Args:
            enhancements (list): List of enhancement names to apply
            classifier (str): Classifier type ('svm', 'rf', or 'nb')

        Returns:
            dict: Model performance metrics
        """
        if self.X_train is None:
            self.prepare_data()

        # Default to all enhancements if none specified
        if enhancements is None:
            enhancements = [
                'text_cleaning', 'sentiment_features', 'text_stats', 'platform_features'
            ]

        enhancement_id = '+'.join(enhancements)

        # Apply enhancements to training data
        X_train_enhanced = self.apply_enhancements(self.X_train, enhancements)

        # Apply enhancements to test data
        X_test_enhanced = self.apply_enhancements(self.X_test, enhancements)

        # Create classifier
        if classifier == 'svm':
            clf = SVC(probability=True)
        elif classifier == 'rf':
            clf = RandomForestClassifier(n_estimators=100)
        elif classifier == 'nb':
            clf = MultinomialNB()
        else:
            raise ValueError(f"Unsupported classifier: {classifier}")

        # Train model
        clf.fit(X_train_enhanced, self.y_train)

        # Evaluate model
        y_pred = clf.predict(X_test_enhanced)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )

        # Store feature importances if possible
        if classifier == 'rf':
            self.feature_importances[enhancement_id] = {
                'feature_names': X_train_enhanced.columns.tolist(),
                'importances': clf.feature_importances_.tolist()
            }

        # Store results
        results = {
            'model': clf,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'enhancements': enhancements
        }

        self.results[enhancement_id] = results

        print(f"Enhanced model ({enhancement_id}):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return results

    def apply_enhancements(self, texts, enhancements):
        """
        Apply specified enhancements to texts.

        Args:
            texts (pd.Series): Texts to enhance
            enhancements (list): List of enhancement names to apply

        Returns:
            pd.DataFrame: Enhanced features
        """
        features = {}

        # Apply text cleaning if specified
        if 'text_cleaning' in enhancements:
            print("Applying text cleaning...")
            cleaned_texts = texts.apply(self.clean_text)

            # Use TF-IDF features of cleaned text
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_features = vectorizer.fit_transform(cleaned_texts)

            # Convert to dataframe
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )

            # Add features to result
            for col in tfidf_df.columns:
                features[col] = tfidf_df[col].values
        else:
            # Use basic TF-IDF features of raw text
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_features = vectorizer.fit_transform(texts)

            # Convert to dataframe
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            )

            # Add features to result
            for col in tfidf_df.columns:
                features[col] = tfidf_df[col].values

        # Apply sentiment features if specified
        if 'sentiment_features' in enhancements:
            print("Extracting sentiment features...")
            sentiment_features = texts.apply(self.extract_sentiment_features)

            # Extract and add features
            features['sentiment_pos'] = sentiment_features.apply(lambda x: x['pos']).values
            features['sentiment_neg'] = sentiment_features.apply(lambda x: x['neg']).values
            features['sentiment_neu'] = sentiment_features.apply(lambda x: x['neu']).values
            features['sentiment_compound'] = sentiment_features.apply(lambda x: x['compound']).values

        # Apply text statistics if specified
        if 'text_stats' in enhancements:
            print("Extracting text statistics...")
            text_stats = texts.apply(self.extract_text_stats)

            # Extract and add features
            features['word_count'] = text_stats.apply(lambda x: x['word_count']).values
            features['avg_word_length'] = text_stats.apply(lambda x: x['avg_word_length']).values
            features['char_count'] = text_stats.apply(lambda x: x['char_count']).values

        # Apply platform features if specified
        if 'platform_features' in enhancements:
            print("Extracting platform features...")
            platform_features = texts.apply(self.extract_platform_features)

            # Extract and add features
            features['has_netflix'] = platform_features.apply(lambda x: x['has_netflix']).values
            features['has_disney'] = platform_features.apply(lambda x: x['has_disney']).values
            features['has_hulu'] = platform_features.apply(lambda x: x['has_hulu']).values
            features['has_amazon'] = platform_features.apply(lambda x: x['has_amazon']).values
            features['has_hbo'] = platform_features.apply(lambda x: x['has_hbo']).values

        # Convert to dataframe
        enhanced_df = pd.DataFrame(features)

        return enhanced_df

    def perform_ablation_study(self, classifier='rf'):
        """
        Perform a comprehensive ablation study.

        Args:
            classifier (str): Classifier type ('svm', 'rf', or 'nb')

        Returns:
            dict: Ablation study results
        """
        print("Starting ablation study...")

        if self.X_train is None:
            self.prepare_data()

        # Define all enhancements
        all_enhancements = [
            'text_cleaning', 'sentiment_features', 'text_stats', 'platform_features'
        ]

        # Train baseline model (no enhancements)
        baseline_results = self.train_baseline_model(classifier)

        # Train model with all enhancements
        all_enhancements_results = self.train_enhanced_model(all_enhancements, classifier)

        # Train models with individual enhancements
        for enhancement in all_enhancements:
            self.train_enhanced_model([enhancement], classifier)

        # Train models with one enhancement removed at a time
        for enhancement in all_enhancements:
            remaining_enhancements = [e for e in all_enhancements if e != enhancement]
            self.train_enhanced_model(remaining_enhancements, classifier)

        # Calculate enhancement contributions
        contributions = {}
        baseline_accuracy = baseline_results['accuracy']
        all_accuracy = all_enhancements_results['accuracy']

        # Individual enhancement contributions
        for enhancement in all_enhancements:
            enhancement_id = enhancement
            enhancement_accuracy = self.results[enhancement_id]['accuracy']

            # Contribution is the improvement over baseline
            contribution = enhancement_accuracy - baseline_accuracy
            contributions[enhancement] = contribution

        # Leave-one-out contributions
        for enhancement in all_enhancements:
            remaining_enhancements = [e for e in all_enhancements if e != enhancement]
            enhancement_id = '+'.join(remaining_enhancements)

            if enhancement_id in self.results:
                without_enhancement_accuracy = self.results[enhancement_id]['accuracy']

                # Contribution is the drop in accuracy when removed
                leave_one_out_contribution = all_accuracy - without_enhancement_accuracy
                contributions[f"{enhancement}_leave_one_out"] = leave_one_out_contribution

        # Store ablation results
        ablation_results = {
            'baseline': baseline_results,
            'all_enhancements': all_enhancements_results,
            'individual_enhancements': {
                enhancement: self.results[enhancement]
                for enhancement in all_enhancements
            },
            'leave_one_out': {
                enhancement: self.results['+'.join([e for e in all_enhancements if e != enhancement])]
                for enhancement in all_enhancements
            },
            'contributions': contributions,
            'feature_importances': self.feature_importances
        }

        print("Ablation study completed.")
        return ablation_results

    def visualize_ablation_results(self, output_dir=None):
        """
        Visualize ablation study results.

        Args:
            output_dir (str, optional): Directory to save visualizations

        Returns:
            dict: Paths to generated visualizations
        """
        if not self.results:
            raise ValueError("No results available. Run ablation study first.")

        if output_dir is None:
            output_dir = 'evaluation/results'

        os.makedirs(output_dir, exist_ok=True)

        visualizations = {}

        # Extract metrics for all models
        model_names = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for model_id, results in self.results.items():
            model_names.append(model_id)
            accuracies.append(results['accuracy'])
            precisions.append(results['precision'])
            recalls.append(results['recall'])
            f1_scores.append(results['f1'])

        # Create accuracy comparison chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, accuracies)

        # Highlight baseline and all enhancements
        bar_colors = ['#1f77b4'] * len(bars)  # Default color
        if 'baseline' in model_names:
            bar_colors[model_names.index('baseline')] = '#d62728'  # Red for baseline
        if 'text_cleaning+sentiment_features+text_stats+platform_features' in model_names:
            bar_colors[model_names.index(
                'text_cleaning+sentiment_features+text_stats+platform_features')] = '#2ca02c'  # Green for all enhancements

        for bar, color in zip(bars, bar_colors):
            bar.set_color(color)

        plt.title('Model Accuracy Comparison')
        plt.xlabel('Model Enhancements')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Save figure
        accuracy_path = os.path.join(output_dir, 'ablation_accuracy.png')
        plt.savefig(accuracy_path)
        plt.close()
        visualizations['accuracy_comparison'] = accuracy_path

        # Create enhancement contribution chart
        if 'baseline' in self.results and 'text_cleaning+sentiment_features+text_stats+platform_features' in self.results:
            # Calculate contributions
            enhancements = ['text_cleaning', 'sentiment_features', 'text_stats', 'platform_features']
            baseline_accuracy = self.results['baseline']['accuracy']

            individual_contributions = []
            for enhancement in enhancements:
                if enhancement in self.results:
                    contrib = self.results[enhancement]['accuracy'] - baseline_accuracy
                    individual_contributions.append(contrib)
                else:
                    individual_contributions.append(0)

            leave_one_out_contributions = []
            all_accuracy = self.results['text_cleaning+sentiment_features+text_stats+platform_features']['accuracy']

            for enhancement in enhancements:
                remaining = '+'.join([e for e in enhancements if e != enhancement])
                if remaining in self.results:
                    contrib = all_accuracy - self.results[remaining]['accuracy']
                    leave_one_out_contributions.append(contrib)
                else:
                    leave_one_out_contributions.append(0)

            # Create figure
            plt.figure(figsize=(12, 6))

            # Set up bar positions
            bar_width = 0.35
            r1 = np.arange(len(enhancements))
            r2 = [x + bar_width for x in r1]

            # Create bars
            plt.bar(r1, individual_contributions, width=bar_width, label='Individual Contribution')
            plt.bar(r2, leave_one_out_contributions, width=bar_width, label='Leave-One-Out Impact')

            # Add labels and legend
            plt.title('Enhancement Contributions to Model Accuracy')
            plt.xlabel('Enhancement')
            plt.ylabel('Contribution to Accuracy')
            plt.xticks([r + bar_width / 2 for r in range(len(enhancements))], enhancements, rotation=45)
            plt.legend()
            plt.tight_layout()

            # Save figure
            contrib_path = os.path.join(output_dir, 'enhancement_contributions.png')
            plt.savefig(contrib_path)
            plt.close()
            visualizations['enhancement_contributions'] = contrib_path

        # Create feature importance visualization for random forest
        if 'text_cleaning+sentiment_features+text_stats+platform_features' in self.feature_importances:
            importances = self.feature_importances['text_cleaning+sentiment_features+text_stats+platform_features']

            # Get non-TF-IDF features
            feature_names = importances['feature_names']
            importance_values = importances['importances']

            # Filter to show only top TF-IDF features and all non-TF-IDF features
            non_tfidf_indices = [i for i, name in enumerate(feature_names) if not name.startswith('tfidf_')]
            tfidf_indices = [i for i, name in enumerate(feature_names) if name.startswith('tfidf_')]

            # Sort TF-IDF features by importance and take top 10
            top_tfidf_indices = sorted(tfidf_indices, key=lambda i: importance_values[i], reverse=True)[:10]

            # Combine non-TF-IDF and top TF-IDF indices
            combined_indices = non_tfidf_indices + top_tfidf_indices

            # Extract feature names and importances
            selected_names = [feature_names[i] for i in combined_indices]
            selected_importances = [importance_values[i] for i in combined_indices]

            # Sort by importance
            sorted_indices = np.argsort(selected_importances)
            sorted_names = [selected_names[i] for i in sorted_indices]
            sorted_importances = [selected_importances[i] for i in sorted_indices]

            # Create figure
            plt.figure(figsize=(12, 8))
            bars = plt.barh(sorted_names, sorted_importances)

            # Color TF-IDF features differently
            colors = ['#1f77b4' if name.startswith('tfidf_') else '#2ca02c' for name in sorted_names]
            for bar, color in zip(bars, colors):
                bar.set_color(color)

            plt.title('Feature Importance in Random Forest Model')
            plt.xlabel('Importance')
            plt.tight_layout()

            # Save figure
            importance_path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(importance_path)
            plt.close()
            visualizations['feature_importance'] = importance_path

        # Create a detailed metrics comparison
        plt.figure(figsize=(12, 8))

        # Set up bar positions
        bar_width = 0.2
        r1 = np.arange(len(model_names))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        r4 = [x + bar_width for x in r3]

        # Create bars
        plt.bar(r1, accuracies, width=bar_width, label='Accuracy')
        plt.bar(r2, precisions, width=bar_width, label='Precision')
        plt.bar(r3, recalls, width=bar_width, label='Recall')
        plt.bar(r4, f1_scores, width=bar_width, label='F1 Score')

        # Add labels and legend
        plt.title('Model Performance Metrics Comparison')
        plt.xlabel('Model Enhancements')
        plt.ylabel('Score')
        plt.xticks([r + 1.5 * bar_width for r in range(len(model_names))], model_names, rotation=90)
        plt.legend()
        plt.tight_layout()

        # Save figure
        metrics_path = os.path.join(output_dir, 'model_metrics_comparison.png')
        plt.savefig(metrics_path)
        plt.close()
        visualizations['metrics_comparison'] = metrics_path

        return visualizations

    def export_results(self, output_path=None):
        """
        Export ablation study results to JSON.

        Args:
            output_path (str, optional): Path to save results

        Returns:
            str: Path to saved results
        """
        if not self.results:
            raise ValueError("No results available. Run ablation study first.")

        if output_path is None:
            output_path = 'evaluation/results/ablation_results.json'

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Prepare results for JSON
        export_results = {}

        for model_id, results in self.results.items():
            export_results[model_id] = {
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1'],
                'enhancements': results['enhancements']
            }

        # Export feature importances
        if self.feature_importances:
            # Export only feature names and their values, not the actual features
            for model_id, importances in self.feature_importances.items():
                # Find top 20 features
                feature_names = importances['feature_names']
                importance_values = importances['importances']

                # Sort by importance
                sorted_indices = np.argsort(importance_values)[::-1][:20]

                # Extract top features
                top_features = {
                    feature_names[i]: importance_values[i] for i in sorted_indices
                }

                export_results[f"{model_id}_top_features"] = top_features

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(export_results, f, indent=2)

        print(f"Results exported to {output_path}")
        return output_path

    def run_complete_ablation_study(self, classifier='rf'):
        """
        Run a complete ablation study and export results.

        Args:
            classifier (str): Classifier type ('svm', 'rf', or 'nb')

        Returns:
            dict: Ablation study results and visualization paths
        """
        if self.X_train is None:
            self.prepare_data()

        # Perform ablation study
        ablation_results = self.perform_ablation_study(classifier)

        # Visualize results
        visualization_paths = self.visualize_ablation_results()

        # Export results
        results_path = self.export_results()

        return {
            'ablation_results': ablation_results,
            'visualization_paths': visualization_paths,
            'results_path': results_path
        }


# Example usage:
if __name__ == "__main__":
    # Path to the dataset
    data_path = "../data/processed_streaming_opinions.csv"

    # Create ablation study
    ablation_study = AblationStudy(data_path=data_path)

    # Load data and prepare for classification
    ablation_study.load_data()
    ablation_study.prepare_data(text_column='text', label_column='sentiment')

    # Run complete ablation study
    results = ablation_study.run_complete_ablation_study(classifier='rf')

    print(f"Ablation study completed. Results saved to {results['results_path']}")
    print("Visualization paths:")
    for name, path in results['visualization_paths'].items():
        print(f"- {name}: {path}")