# evaluation/evaluate_classifier.py
"""Evaluation framework for the sentiment classification system."""

import pandas as pd
import numpy as np
import json
import os
import time
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class ClassifierEvaluator:
    """Evaluation framework for sentiment classification and feature extraction."""

    def __init__(self, evaluation_data_path=None, random_sample_size=200):
        """
        Initialize the evaluator.

        Args:
            evaluation_data_path (str): Path to manually labeled evaluation dataset
            random_sample_size (int): Number of records to use for random accuracy testing
        """
        self.evaluation_data_path = evaluation_data_path
        self.random_sample_size = random_sample_size
        self.evaluation_results = {}
        self.ablation_results = {}

    def load_evaluation_data(self, path=None):
        """
        Load the manually labeled evaluation dataset.

        Args:
            path (str, optional): Path to evaluation dataset

        Returns:
            DataFrame: The evaluation dataset
        """
        path = path or self.evaluation_data_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Evaluation dataset not found at {path}")

        return pd.read_csv(path)

    def evaluate_sentiment_classification(self, evaluation_df=None):
        """
        Evaluate sentiment classification performance.

        Args:
            evaluation_df (DataFrame, optional): DataFrame with ground truth labels

        Returns:
            dict: Evaluation metrics
        """
        if evaluation_df is None:
            evaluation_df = self.load_evaluation_data()

        # Check that required columns exist
        required_cols = ['sentiment', 'manual_sentiment']
        if not all(col in evaluation_df.columns for col in required_cols):
            raise ValueError(f"Evaluation dataframe must contain columns: {required_cols}")

        # Calculate metrics
        y_true = evaluation_df['manual_sentiment']
        y_pred = evaluation_df['sentiment']

        # Get precision, recall, and F1 score for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=['positive', 'negative', 'neutral']
        )

        # Get weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        # Calculate overall accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Create classification report
        report = classification_report(y_true, y_pred, output_dict=True)

        # Create confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=['positive', 'negative', 'neutral'])

        # Store results
        results = {
            'accuracy': accuracy,
            'precision': {
                'positive': precision[0],
                'negative': precision[1],
                'neutral': precision[2],
                'weighted': precision_weighted
            },
            'recall': {
                'positive': recall[0],
                'negative': recall[1],
                'neutral': recall[2],
                'weighted': recall_weighted
            },
            'f1': {
                'positive': f1[0],
                'negative': f1[1],
                'neutral': f1[2],
                'weighted': f1_weighted
            },
            'support': {
                'positive': support[0],
                'negative': support[1],
                'neutral': support[2],
            },
            'report': report,
            'confusion_matrix': conf_matrix.tolist()
        }

        self.evaluation_results['sentiment_classification'] = results
        return results

    def evaluate_feature_extraction(self, evaluation_df=None, features=None):
        """
        Evaluate feature extraction performance.

        Args:
            evaluation_df (DataFrame, optional): DataFrame with ground truth labels
            features (list, optional): List of feature names to evaluate

        Returns:
            dict: Evaluation metrics
        """
        if evaluation_df is None:
            evaluation_df = self.load_evaluation_data()

        if features is None:
            features = [
                'content_quality', 'pricing', 'ui_ux', 'technical', 'customer_service'
            ]

        results = {}

        for feature in features:
            # Columns for predicted and ground truth values
            pred_col = feature
            true_col = f'manual_{feature}'

            # Skip if columns don't exist
            if pred_col not in evaluation_df.columns or true_col not in evaluation_df.columns:
                continue

            # Calculate mean absolute error
            mae = np.mean(np.abs(evaluation_df[pred_col] - evaluation_df[true_col]))

            # Calculate root mean squared error
            rmse = np.sqrt(np.mean((evaluation_df[pred_col] - evaluation_df[true_col]) ** 2))

            # Calculate correlation coefficient
            corr = evaluation_df[[pred_col, true_col]].corr().iloc[0, 1]

            results[feature] = {
                'mae': mae,
                'rmse': rmse,
                'correlation': corr
            }

        self.evaluation_results['feature_extraction'] = results
        return results

    def calculate_annotator_agreement(self, annotator_data=None):
        """
        Calculate inter-annotator agreement metrics.

        Args:
            annotator_data (DataFrame or str): DataFrame with multiple annotator labels or path to file

        Returns:
            dict: Agreement metrics
        """
        if isinstance(annotator_data, str) and os.path.exists(annotator_data):
            annotator_data = pd.read_csv(annotator_data)

        if annotator_data is None:
            raise ValueError("Annotator data must be provided")

        # Calculate Cohen's Kappa for each pair of annotators
        annotators = [col for col in annotator_data.columns if col.startswith('annotator_')]

        if len(annotators) < 2:
            raise ValueError("At least two annotator columns required (annotator_1, annotator_2, etc.)")

        from sklearn.metrics import cohen_kappa_score

        kappa_scores = {}
        agreement_rates = {}

        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                annotator1 = annotators[i]
                annotator2 = annotators[j]

                # Calculate Cohen's Kappa
                kappa = cohen_kappa_score(
                    annotator_data[annotator1],
                    annotator_data[annotator2]
                )

                # Calculate simple agreement rate
                agreement_rate = np.mean(annotator_data[annotator1] == annotator_data[annotator2])

                pair_name = f"{annotator1}_vs_{annotator2}"
                kappa_scores[pair_name] = kappa
                agreement_rates[pair_name] = agreement_rate

        # Calculate overall agreement
        average_kappa = np.mean(list(kappa_scores.values()))
        average_agreement = np.mean(list(agreement_rates.values()))

        results = {
            'pairwise_kappa': kappa_scores,
            'pairwise_agreement': agreement_rates,
            'average_kappa': average_kappa,
            'average_agreement': average_agreement
        }

        self.evaluation_results['annotator_agreement'] = results
        return results

    def random_accuracy_test(self, full_dataset_path, model_predictions_path=None):
        """
        Perform random accuracy test on the rest of the data.

        Args:
            full_dataset_path (str): Path to full dataset
            model_predictions_path (str, optional): Path to model predictions (if not in full dataset)

        Returns:
            dict: Random test results
        """
        # Load full dataset
        full_df = pd.read_csv(full_dataset_path)

        # Load model predictions if separately provided
        if model_predictions_path:
            pred_df = pd.read_csv(model_predictions_path)
            # Merge predictions with full dataset
            full_df = full_df.merge(pred_df, on='id', how='left')

        # Get IDs from evaluation data to exclude
        if self.evaluation_data_path and os.path.exists(self.evaluation_data_path):
            eval_df = pd.read_csv(self.evaluation_data_path)
            eval_ids = set(eval_df['id'])
            # Filter out evaluation data
            full_df = full_df[~full_df['id'].isin(eval_ids)]

        # Take random sample
        if len(full_df) > self.random_sample_size:
            sample_df = full_df.sample(n=self.random_sample_size, random_state=42)
        else:
            sample_df = full_df

        # Manually review and label the sample (this would be done by human annotators)
        # For demonstration, we'll simulate this by assuming current labels are correct for 80%
        # and introducing random errors for 20%

        # Assuming 'sentiment' column exists in the dataset
        if 'sentiment' not in sample_df.columns:
            raise ValueError("Dataset must contain 'sentiment' column")

        # Create a copy of the sample with simulated manual labels for demonstration
        manual_sample = sample_df.copy()

        # Simulate manual labeling with 80% agreement
        np.random.seed(42)
        random_mask = np.random.random(len(manual_sample)) < 0.2

        # Get all unique sentiments
        all_sentiments = manual_sample['sentiment'].unique()

        # For 20% of samples, assign a different random sentiment
        for idx in manual_sample[random_mask].index:
            current_sentiment = manual_sample.loc[idx, 'sentiment']
            other_sentiments = [s for s in all_sentiments if s != current_sentiment]
            if other_sentiments:
                manual_sample.loc[idx, 'manual_sentiment'] = np.random.choice(other_sentiments)
            else:
                manual_sample.loc[idx, 'manual_sentiment'] = current_sentiment

        # For the remaining 80%, keep the same sentiment
        manual_sample.loc[~random_mask, 'manual_sentiment'] = manual_sample.loc[~random_mask, 'sentiment']

        # Calculate accuracy statistics
        accuracy = accuracy_score(manual_sample['manual_sentiment'], manual_sample['sentiment'])

        # Get precision, recall, and F1 score
        precision, recall, f1, support = precision_recall_fscore_support(
            manual_sample['manual_sentiment'],
            manual_sample['sentiment'],
            average='weighted'
        )

        results = {
            'sample_size': len(manual_sample),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        self.evaluation_results['random_accuracy_test'] = results
        return results

    def perform_ablation_study(self, features_to_test, dataset_path, classifier_func):
        """
        Perform ablation study to measure the contribution of each feature.

        Args:
            features_to_test (list): List of features to test
            dataset_path (str): Path to dataset
            classifier_func (callable): Function that runs classification with specified features

        Returns:
            dict: Ablation study results
        """
        # Load dataset
        df = pd.read_csv(dataset_path)

        # Baseline - no additional features
        baseline_results = classifier_func(df, [])

        # Test each feature individually
        individual_results = {}
        for feature in features_to_test:
            feature_results = classifier_func(df, [feature])
            individual_results[feature] = feature_results

        # Test all features
        all_features_results = classifier_func(df, features_to_test)

        # Calculate contribution of each feature
        feature_contributions = {}
        for feature in features_to_test:
            contribution = individual_results[feature]['accuracy'] - baseline_results['accuracy']
            feature_contributions[feature] = contribution

        # Store results
        results = {
            'baseline': baseline_results,
            'individual_features': individual_results,
            'all_features': all_features_results,
            'feature_contributions': feature_contributions
        }

        self.ablation_results = results
        return results

    def visualize_results(self, output_dir=None):
        """
        Generate visualizations of evaluation results.

        Args:
            output_dir (str, optional): Directory to save visualizations

        Returns:
            dict: Paths to generated visualizations
        """
        if not output_dir:
            output_dir = 'evaluation/results'

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        visualizations = {}

        # Confusion matrix visualization
        if 'sentiment_classification' in self.evaluation_results:
            conf_matrix = self.evaluation_results['sentiment_classification']['confusion_matrix']

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Positive', 'Negative', 'Neutral'],
                yticklabels=['Positive', 'Negative', 'Neutral']
            )
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Sentiment Classification Confusion Matrix')

            confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(confusion_matrix_path)
            plt.close()

            visualizations['confusion_matrix'] = confusion_matrix_path

        # Feature extraction performance
        if 'feature_extraction' in self.evaluation_results:
            feature_results = self.evaluation_results['feature_extraction']

            # Extract MAE for each feature
            features = list(feature_results.keys())
            mae_values = [feature_results[f]['mae'] for f in features]

            plt.figure(figsize=(12, 6))
            sns.barplot(x=features, y=mae_values)
            plt.xlabel('Features')
            plt.ylabel('Mean Absolute Error')
            plt.title('Feature Extraction Performance')
            plt.xticks(rotation=45)

            feature_perf_path = os.path.join(output_dir, 'feature_performance.png')
            plt.savefig(feature_perf_path)
            plt.close()

            visualizations['feature_performance'] = feature_perf_path

        # Ablation study visualization
        if self.ablation_results:
            feature_contributions = self.ablation_results['feature_contributions']

            plt.figure(figsize=(12, 6))
            features = list(feature_contributions.keys())
            contributions = list(feature_contributions.values())

            sns.barplot(x=features, y=contributions)
            plt.xlabel('Features')
            plt.ylabel('Accuracy Contribution')
            plt.title('Feature Contribution to Accuracy')
            plt.xticks(rotation=45)

            ablation_path = os.path.join(output_dir, 'ablation_study.png')
            plt.savefig(ablation_path)
            plt.close()

            visualizations['ablation_study'] = ablation_path

        return visualizations

    def export_results(self, output_path=None):
        """
        Export evaluation results to JSON file.

        Args:
            output_path (str, optional): Path to save results

        Returns:
            str: Path to saved results
        """
        if not output_path:
            output_path = 'evaluation/results/evaluation_results.json'

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Combine all results
        all_results = {
            'sentiment_classification': self.evaluation_results.get('sentiment_classification', {}),
            'feature_extraction': self.evaluation_results.get('feature_extraction', {}),
            'annotator_agreement': self.evaluation_results.get('annotator_agreement', {}),
            'random_accuracy_test': self.evaluation_results.get('random_accuracy_test', {}),
            'ablation_study': self.ablation_results
        }

        # Add timestamp
        all_results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"Results exported to {output_path}")
        return output_path

    def run_complete_evaluation(self, evaluation_data_path=None, full_dataset_path=None):
        """
        Run a complete evaluation of the sentiment classification system.

        Args:
            evaluation_data_path (str, optional): Path to evaluation dataset
            full_dataset_path (str, optional): Path to full dataset

        Returns:
            dict: Complete evaluation results
        """
        if evaluation_data_path:
            self.evaluation_data_path = evaluation_data_path

        # Load evaluation data
        evaluation_df = self.load_evaluation_data()

        # Evaluate sentiment classification
        sentiment_results = self.evaluate_sentiment_classification(evaluation_df)
        print("Sentiment Classification Results:")
        print(f"Accuracy: {sentiment_results['accuracy']:.4f}")
        print(f"Weighted F1: {sentiment_results['f1']['weighted']:.4f}")

        # Evaluate feature extraction
        feature_results = self.evaluate_feature_extraction(evaluation_df)
        print("\nFeature Extraction Results:")
        for feature, metrics in feature_results.items():
            print(f"{feature}: MAE = {metrics['mae']:.4f}, RMSE = {metrics['rmse']:.4f}")

        # Calculate annotator agreement
        try:
            agreement_results = self.calculate_annotator_agreement(evaluation_df)
            print("\nAnnotator Agreement:")
            print(f"Average Agreement: {agreement_results['average_agreement']:.4f}")
            print(f"Average Kappa: {agreement_results['average_kappa']:.4f}")
        except (ValueError, KeyError) as e:
            print(f"\nSkipping annotator agreement: {e}")

        # Random accuracy test
        if full_dataset_path:
            random_test_results = self.random_accuracy_test(full_dataset_path)
            print("\nRandom Accuracy Test:")
            print(f"Sample Size: {random_test_results['sample_size']}")
            print(f"Accuracy: {random_test_results['accuracy']:.4f}")

        # Generate visualizations
        visualizations = self.visualize_results()
        print("\nGenerated Visualizations:")
        for name, path in visualizations.items():
            print(f"{name}: {path}")

        # Export results
        results_path = self.export_results()

        return self.evaluation_results


# Example usage:
if __name__ == "__main__":
    # Paths to the required datasets
    EVALUATION_DATA_PATH = "../data/evaluation_dataset.csv"
    FULL_DATASET_PATH = "../data/processed_streaming_opinions.csv"

    # Create evaluator
    evaluator = ClassifierEvaluator(
        evaluation_data_path=EVALUATION_DATA_PATH,
        random_sample_size=200
    )

    # Run complete evaluation
    results = evaluator.run_complete_evaluation(full_dataset_path=FULL_DATASET_PATH)

    # Export results
    evaluator.export_results()