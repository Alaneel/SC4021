"""
Evaluation and annotation tools for the classifier
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import json
import os
from datetime import datetime
import sys

from config.app_config import EVALUATION_DIR
from .classifier import EVOpinionClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class EVClassifierEvaluator:
    """
    Evaluator for the EV Opinion Classifier
    """

    def __init__(self, classifier=None):
        """
        Initialize the evaluator

        Args:
            classifier (EVOpinionClassifier): Classifier instance to evaluate
        """
        self.classifier = classifier or EVOpinionClassifier()

        # Make sure we have the models loaded
        if not hasattr(self.classifier, 'sentiment_pipeline') or self.classifier.sentiment_pipeline is None:
            self.classifier.load_sentiment_model()

    def evaluate_sentiment(self, df_test, ground_truth_col='manual_sentiment', text_col='text'):
        """
        Evaluate sentiment analysis against ground truth

        Args:
            df_test (pandas.DataFrame): Test dataframe
            ground_truth_col (str): Column name with ground truth
            text_col (str): Column name with text content

        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating sentiment analysis on {len(df_test)} samples")

        predictions = []
        ground_truth = []

        for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Evaluating sentiment"):
            text = row[text_col]
            true_sentiment = row[ground_truth_col]

            if pd.isna(true_sentiment):
                continue

            sentiment_result = self.classifier.analyze_sentiment(text)
            predicted_sentiment = sentiment_result['label']

            predictions.append(predicted_sentiment)
            ground_truth.append(true_sentiment)

        # Calculate metrics
        logger.info("\nSentiment Analysis Evaluation:")
        report = classification_report(ground_truth, predictions, output_dict=True)
        logger.info(f"\n{classification_report(ground_truth, predictions)}")

        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions,
                              labels=['positive', 'neutral', 'negative'])

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['positive', 'neutral', 'negative'],
                    yticklabels=['positive', 'neutral', 'negative'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Sentiment Analysis Confusion Matrix')
        plt.tight_layout()

        # Save the figure
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        cm_path = os.path.join(EVALUATION_DIR, 'sentiment_confusion_matrix.png')
        plt.savefig(cm_path)
        logger.info(f"Saved confusion matrix to {cm_path}")

        # Calculate overall metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='weighted'
        )

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        # Create results dictionary
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'num_samples': len(ground_truth)
        }

        # Save results to JSON
        results_path = os.path.join(EVALUATION_DIR, 'sentiment_evaluation.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Saved evaluation results to {results_path}")

        return results

    def perform_ablation_study(self, df_train, df_test, output_path=None):
        """
        Perform ablation study to measure impact of different enhancements

        Args:
            df_train (pandas.DataFrame): Training data
            df_test (pandas.DataFrame): Test data
            output_path (str): Path to save study results

        Returns:
            dict: Study results
        """
        logger.info("Starting ablation study...")

        if output_path is None:
            output_path = os.path.join(EVALUATION_DIR, 'ablation_study.json')

        results = {}

        # 1. Baseline: DistilBERT sentiment only
        logger.info("\n1. Baseline: DistilBERT sentiment only")
        classifier_baseline = EVOpinionClassifier(sentiment_model="distilbert-base-uncased-finetuned-sst-2-english")
        classifier_baseline.load_sentiment_model()
        evaluator_baseline = EVClassifierEvaluator(classifier_baseline)
        baseline_results = evaluator_baseline.evaluate_sentiment(df_test)
        results['baseline'] = {
            'accuracy': baseline_results['accuracy'],
            'precision': baseline_results['precision'],
            'recall': baseline_results['recall'],
            'f1': baseline_results['f1']
        }

        # 2. Enhancement 1: RoBERTa sentiment model
        logger.info("\n2. Enhancement 1: RoBERTa sentiment model")
        classifier_roberta = EVOpinionClassifier(sentiment_model="siebert/sentiment-roberta-large-english")
        classifier_roberta.load_sentiment_model()
        evaluator_roberta = EVClassifierEvaluator(classifier_roberta)
        roberta_results = evaluator_roberta.evaluate_sentiment(df_test)
        results['roberta'] = {
            'accuracy': roberta_results['accuracy'],
            'precision': roberta_results['precision'],
            'recall': roberta_results['recall'],
            'f1': roberta_results['f1']
        }

        # 3. Enhancement 2: Topic filtering
        logger.info("\n3. Enhancement 2: Topic modeling + filtering")
        classifier_topics = EVOpinionClassifier()
        classifier_topics.load_sentiment_model()
        classifier_topics.build_topic_model(df_train['text'].tolist(), num_topics=15)

        # Process test data with topic modeling
        logger.info("Assigning topics to test data...")
        test_with_topics = classifier_topics.process_data(
            df_test,
            analyze_sentiment=False,
            extract_entities=False,
            assign_topics=True
        )

        # Define EV-related keywords for filtering
        ev_keywords = ['tesla', 'electric', 'battery', 'charge', 'ev', 'vehicle', 'car', 'motor']

        # Filter to only keep samples with EV-related topics
        def has_ev_topic(topics):
            if pd.isna(topics) or not topics:
                return False

            # Convert string representation to list if needed
            if isinstance(topics, str):
                try:
                    topics = eval(topics)
                except:
                    return False

            if not topics:
                return False

            # Check if any topic contains EV keywords
            for topic in topics:
                for keyword in ev_keywords:
                    if keyword in topic.lower():
                        return True
            return False

        # Apply filter
        test_filtered = test_with_topics[test_with_topics['topics'].apply(has_ev_topic)]
        logger.info(f"Filtered to {len(test_filtered)} samples with EV-related topics")

        # Evaluate
        evaluator_topics = EVClassifierEvaluator(classifier_topics)
        topic_results = evaluator_topics.evaluate_sentiment(test_filtered)
        results['topic_filtered'] = {
            'accuracy': topic_results['accuracy'],
            'precision': topic_results['precision'],
            'recall': topic_results['recall'],
            'f1': topic_results['f1']
        }

        # 4. Enhancement 3: Entity recognition
        logger.info("\n4. Enhancement 3: Entity recognition")
        classifier_entities = EVOpinionClassifier()
        classifier_entities.load_sentiment_model()

        # Process test data with entity extraction
        logger.info("Extracting entities from test data...")
        test_with_entities = classifier_entities.process_data(
            df_test,
            analyze_sentiment=False,
            extract_entities=True,
            assign_topics=False
        )

        # Filter to samples with EV-related entities
        def has_ev_entity(entities):
            if pd.isna(entities) or not entities:
                return False

            # Convert string representation to list if needed
            if isinstance(entities, str):
                try:
                    entities = eval(entities)
                except:
                    return False

            if not entities:
                return False

            # Check if any entity is EV-related
            for entity in entities:
                for keyword in ev_keywords:
                    if keyword in entity.lower():
                        return True
            return False

        # Apply filter
        entity_filtered = test_with_entities[test_with_entities['entities'].apply(has_ev_entity)]
        logger.info(f"Filtered to {len(entity_filtered)} samples with EV-related entities")

        # Evaluate
        evaluator_entities = EVClassifierEvaluator(classifier_entities)
        entity_results = evaluator_entities.evaluate_sentiment(entity_filtered)
        results['entity_filtered'] = {
            'accuracy': entity_results['accuracy'],
            'precision': entity_results['precision'],
            'recall': entity_results['recall'],
            'f1': entity_results['f1']
        }

        # 5. All enhancements combined
        logger.info("\n5. All enhancements combined")
        classifier_combined = EVOpinionClassifier(sentiment_model="siebert/sentiment-roberta-large-english")
        classifier_combined.load_sentiment_model()
        classifier_combined.build_topic_model(df_train['text'].tolist(), num_topics=15)

        # Process with all enhancements
        logger.info("Processing test data with all enhancements...")
        combined_test = classifier_combined.process_data(
            df_test,
            analyze_sentiment=False,
            extract_entities=True,
            assign_topics=True
        )

        # Filter by topics and entities
        combined_filtered = combined_test[
            combined_test['topics'].apply(has_ev_topic) |
            combined_test['entities'].apply(has_ev_entity)
            ]
        logger.info(f"Filtered to {len(combined_filtered)} samples with EV-related content")

        # Evaluate combined approach
        evaluator_combined = EVClassifierEvaluator(classifier_combined)
        combined_results = evaluator_combined.evaluate_sentiment(combined_filtered)
        results['combined'] = {
            'accuracy': combined_results['accuracy'],
            'precision': combined_results['precision'],
            'recall': combined_results['recall'],
            'f1': combined_results['f1']
        }

        # Calculate improvements over baseline
        for enhancement, metrics in results.items():
            if enhancement != 'baseline':
                acc_improvement = metrics['accuracy'] - results['baseline']['accuracy']
                f1_improvement = metrics['f1'] - results['baseline']['f1']

                logger.info(f"\n{enhancement} improvement over baseline:")
                logger.info(
                    f"Accuracy: +{acc_improvement:.4f} ({acc_improvement / results['baseline']['accuracy'] * 100:.1f}%)")
                logger.info(
                    f"F1 Score: +{f1_improvement:.4f} ({f1_improvement / results['baseline']['f1'] * 100:.1f}%)")

                # Add improvement data to results
                results[enhancement]['improvement'] = {
                    'accuracy_abs': acc_improvement,
                    'accuracy_pct': acc_improvement / results['baseline']['accuracy'] * 100,
                    'f1_abs': f1_improvement,
                    'f1_pct': f1_improvement / results['baseline']['f1'] * 100
                }

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"\nAblation study results saved to {output_path}")

        # Create visualization of improvements
        self._plot_ablation_results(results, os.path.dirname(output_path))

        return results

    def _plot_ablation_results(self, results, output_dir):
        """
        Create visualizations of ablation study results

        Args:
            results (dict): Ablation study results
            output_dir (str): Directory to save visualizations
        """
        # Extract F1 scores
        models = list(results.keys())
        f1_scores = [results[model]['f1'] for model in models]

        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, f1_scores, color=['#777777' if m == 'baseline' else '#1E88E5' for m in models])

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom')

        plt.ylim(0, 1.0)
        plt.ylabel('F1 Score')
        plt.title('Ablation Study: F1 Score Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(output_dir, 'ablation_f1_scores.png'))

        # Create comparison of all metrics
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']

        # Prepare data for grouped bar chart
        metric_values = {metric: [] for metric in metrics}
        for model in models:
            for metric in metrics:
                metric_values[metric].append(results[model][metric])

        # Create figure
        plt.figure(figsize=(14, 7))

        # Set width of bars
        bar_width = 0.2
        index = np.arange(len(models))

        # Create bars
        colors = ['#1E88E5', '#FFC107', '#4CAF50', '#F44336']
        for i, metric in enumerate(metrics):
            plt.bar(index + i * bar_width, metric_values[metric], bar_width,
                    label=metric.capitalize(), color=colors[i])

        plt.xlabel('Model Configuration')
        plt.ylabel('Score')
        plt.title('Ablation Study: All Metrics Comparison')
        plt.xticks(index + bar_width * 1.5, models, rotation=45)
        plt.legend()
        plt.ylim(0, 1.0)
        plt.tight_layout()

        # Save the figure
        plt.savefig(os.path.join(output_dir, 'ablation_all_metrics.png'))


def create_annotation_tool(input_df, output_path, num_samples=1000, random_seed=42):
    """
    Create a dataset for manual annotation

    Args:
        input_df (pandas.DataFrame): Input dataframe
        output_path (str): Path to save the annotation template
        num_samples (int): Number of samples to annotate
        random_seed (int): Random seed for reproducibility

    Returns:
        pandas.DataFrame: DataFrame prepared for annotation
    """
    logger.info(f"Creating annotation dataset with {num_samples} samples")

    # Initialize classifier for automatic sentiment analysis
    classifier = EVOpinionClassifier()
    classifier.load_sentiment_model()

    # Check if we already have a partially annotated file
    if os.path.exists(output_path):
        logger.info(f"Found existing annotations at {output_path}, continuing...")
        annotated_df = pd.read_csv(output_path)

        # Get IDs that have already been annotated
        if 'id' in annotated_df.columns and 'manual_sentiment' in annotated_df.columns:
            annotated_ids = set(annotated_df[~pd.isna(annotated_df['manual_sentiment'])]['id'])
            logger.info(f"Already annotated {len(annotated_ids)} samples")

            # Filter out already annotated samples
            input_df = input_df[~input_df['id'].isin(annotated_ids)]

        # Check if we have enough samples left
        if len(input_df) < num_samples:
            logger.warning(f"Only {len(input_df)} unannotated samples left, using all of them")
            num_samples = len(input_df)

    # Sample random rows for annotation
    np.random.seed(random_seed)
    sample_df = input_df.sample(num_samples)

    # Add annotation columns
    sample_df['auto_sentiment'] = None
    sample_df['auto_sentiment_score'] = None
    sample_df['manual_sentiment'] = None
    sample_df['annotation_notes'] = None
    sample_df['annotation_date'] = None

    # Add automatic sentiment as a reference
    logger.info("Adding automatic sentiment analysis...")
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        result = classifier.analyze_sentiment(row['text'])
        sample_df.at[idx, 'auto_sentiment'] = result['label']
        sample_df.at[idx, 'auto_sentiment_score'] = result['score']

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_df.to_csv(output_path, index=False)
    logger.info(f"Saved annotation template to {output_path}")

    # Display statistics
    sentiment_dist = sample_df['auto_sentiment'].value_counts()
    logger.info("\nAutomatic sentiment distribution in annotation set:")
    for sentiment, count in sentiment_dist.items():
        logger.info(f"- {sentiment}: {count} ({count / len(sample_df) * 100:.1f}%)")

    return sample_df


def calculate_interannotator_agreement(file1, file2, id_column='id', sentiment_column='manual_sentiment'):
    """
    Calculate agreement between two annotators

    Args:
        file1 (str): Path to first annotator's CSV file
        file2 (str): Path to second annotator's CSV file
        id_column (str): Column name for the sample ID
        sentiment_column (str): Column name for the sentiment

    Returns:
        float: Cohen's Kappa score
    """
    logger.info(f"Calculating agreement between {file1} and {file2}")

    # Load annotation files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Filter to only include rows with annotations
    df1 = df1[~pd.isna(df1[sentiment_column])]
    df2 = df2[~pd.isna(df2[sentiment_column])]

    # Merge on ID
    merged_df = pd.merge(
        df1[[id_column, sentiment_column]],
        df2[[id_column, sentiment_column]],
        on=id_column,
        suffixes=('_1', '_2')
    )

    logger.info(f"Found {len(merged_df)} samples with annotations from both annotators")

    if len(merged_df) == 0:
        logger.warning("No overlapping annotations found. Cannot calculate agreement.")
        return None

    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(
        merged_df[f'{sentiment_column}_1'],
        merged_df[f'{sentiment_column}_2']
    )

    logger.info(f"Inter-annotator agreement (Cohen's Kappa): {kappa:.4f}")

    # Print disagreement cases
    disagreements = merged_df[merged_df[f'{sentiment_column}_1'] != merged_df[f'{sentiment_column}_2']]
    logger.info(f"Number of disagreements: {len(disagreements)} out of {len(merged_df)} samples "
                f"({len(disagreements) / len(merged_df) * 100:.1f}%)")

    # Print confusion matrix
    cm = confusion_matrix(
        merged_df[f'{sentiment_column}_1'],
        merged_df[f'{sentiment_column}_2'],
        labels=['positive', 'neutral', 'negative']
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['positive', 'neutral', 'negative'],
                yticklabels=['positive', 'neutral', 'negative'])
    plt.ylabel('Annotator 1')
    plt.xlabel('Annotator 2')
    plt.title('Inter-annotator Agreement Confusion Matrix')
    plt.tight_layout()

    # Save the figure
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    output_path = os.path.join(EVALUATION_DIR, 'annotation_agreement.png')
    plt.savefig(output_path)
    logger.info(f"Saved agreement confusion matrix to {output_path}")

    # Save detailed results
    agreement_results = {
        'kappa': float(kappa),
        'num_samples': int(len(merged_df)),
        'num_disagreements': int(len(disagreements)),
        'disagreement_rate': float(len(disagreements) / len(merged_df)),
        'confusion_matrix': cm.tolist()
    }

    results_path = os.path.join(EVALUATION_DIR, 'annotation_agreement.json')
    with open(results_path, 'w') as f:
        json.dump(agreement_results, f, indent=4)
    logger.info(f"Saved agreement results to {results_path}")

    return kappa