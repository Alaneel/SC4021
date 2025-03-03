#!/usr/bin/env python
"""
Script to run the classifier evaluation and annotation tools
"""
import argparse
import os
import sys
import logging
import pandas as pd
import glob

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification import EVOpinionClassifier, EVClassifierEvaluator, create_annotation_tool
from classification.evaluation import calculate_interannotator_agreement
from config.app_config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    EVALUATION_DIR,
    MODELS_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run classifier evaluation and annotation tools')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Annotate command
    annotate_parser = subparsers.add_parser('annotate', help='Create annotation dataset')
    annotate_parser.add_argument('--input', help='Input CSV file path (required if not using --latest)')
    annotate_parser.add_argument('--latest', action='store_true', help='Use the latest data file')
    annotate_parser.add_argument('--output', help='Output CSV file path for annotations')
    annotate_parser.add_argument('--samples', type=int, default=1000,
                                 help='Number of samples to annotate (default: 1000)')

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate classifier on annotated data')
    evaluate_parser.add_argument('--input', required=True, help='Input CSV file with manual annotations')
    evaluate_parser.add_argument('--model-path', help='Path to pre-trained models (default: auto-detect)')
    evaluate_parser.add_argument('--output', help='Output path for evaluation results')

    # Agreement command
    agreement_parser = subparsers.add_parser('agreement', help='Calculate inter-annotator agreement')
    agreement_parser.add_argument('--file1', required=True, help='First annotator file')
    agreement_parser.add_argument('--file2', required=True, help='Second annotator file')

    # Ablation command
    ablation_parser = subparsers.add_parser('ablation', help='Perform ablation study')
    ablation_parser.add_argument('--train', required=True, help='Training data CSV file')
    ablation_parser.add_argument('--test', required=True, help='Test data CSV file with annotations')
    ablation_parser.add_argument('--output', help='Output path for ablation results')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train classifier models')
    train_parser.add_argument('--input', required=True, help='Input CSV file with annotations for training')
    train_parser.add_argument('--output', help='Output path prefix for trained models')
    train_parser.add_argument('--topics', type=int, default=15, help='Number of topics for topic model (default: 15)')

    args = parser.parse_args()

    if args.command == 'annotate':
        run_annotate(args)
    elif args.command == 'evaluate':
        run_evaluate(args)
    elif args.command == 'agreement':
        run_agreement(args)
    elif args.command == 'ablation':
        run_ablation(args)
    elif args.command == 'train':
        run_train(args)
    else:
        parser.print_help()
        return 1

    return 0


def run_annotate(args):
    """Run annotation tool to create labeled dataset"""
    # Find input file
    input_file = args.input

    if not input_file and args.latest:
        # Check processed directory first
        processed_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, "processed_*.csv"))
        raw_files = glob.glob(os.path.join(RAW_DATA_DIR, "reddit_*.csv"))

        if processed_files:
            input_file = max(processed_files, key=os.path.getmtime)
            logger.info(f"Using latest processed file: {input_file}")
        elif raw_files:
            input_file = max(raw_files, key=os.path.getmtime)
            logger.info(f"Using latest raw file: {input_file}")
        else:
            logger.error("No data files found in processed or raw directories")
            return 1

    if not input_file or not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return 1

    # Set output file
    if not args.output:
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        output_file = os.path.join(EVALUATION_DIR, "ev_opinions_annotation.csv")
    else:
        output_file = args.output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load data
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    # Create annotation dataset
    logger.info(f"Creating annotation dataset with {args.samples} samples")
    create_annotation_tool(df, output_file, num_samples=args.samples)

    logger.info(f"Annotation template saved to {output_file}")
    logger.info("Please manually annotate the 'manual_sentiment' column with 'positive', 'negative', or 'neutral'")

    return 0


def run_evaluate(args):
    """Run evaluation on annotated data"""
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Set output file
    if not args.output:
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        output_file = os.path.join(EVALUATION_DIR, "classifier_evaluation.json")
    else:
        output_file = args.output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load data
    logger.info(f"Loading annotated data from {args.input}")
    df = pd.read_csv(args.input)

    # Filter to only include rows with manual annotations
    df = df[~pd.isna(df['manual_sentiment'])]
    logger.info(f"Found {len(df)} samples with manual annotations")

    if len(df) == 0:
        logger.error("No manual annotations found in the input file")
        return 1

    # Initialize classifier
    classifier = EVOpinionClassifier()

    # Try to load pre-trained models
    if args.model_path:
        classifier.load_models(args.model_path)
    else:
        # Auto-detect models in the models directory
        model_files = glob.glob(os.path.join(MODELS_DIR, "ev_classifier_*.model"))
        if model_files:
            model_path = os.path.join(MODELS_DIR, "ev_classifier")
            logger.info(f"Auto-detected model files at {model_path}")
            classifier.load_models(model_path)

    # Initialize evaluator
    evaluator = EVClassifierEvaluator(classifier)

    # Run evaluation
    logger.info("Evaluating classifier on annotated data...")
    results = evaluator.evaluate_sentiment(df, ground_truth_col='manual_sentiment')

    logger.info(f"Evaluation results saved to {output_file}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall: {results['recall']:.4f}")
    logger.info(f"F1 Score: {results['f1']:.4f}")

    return 0


def run_agreement(args):
    """Calculate inter-annotator agreement"""
    # Validate input files
    if not os.path.exists(args.file1):
        logger.error(f"First annotator file not found: {args.file1}")
        return 1

    if not os.path.exists(args.file2):
        logger.error(f"Second annotator file not found: {args.file2}")
        return 1

    # Calculate agreement
    logger.info(f"Calculating agreement between {args.file1} and {args.file2}...")
    kappa = calculate_interannotator_agreement(args.file1, args.file2)

    if kappa is None:
        logger.error("Failed to calculate agreement")
        return 1

    logger.info(f"Inter-annotator agreement (Cohen's Kappa): {kappa:.4f}")
    logger.info(f"Agreement visualization saved to {os.path.join(EVALUATION_DIR, 'annotation_agreement.png')}")

    return 0


def run_ablation(args):
    """Run ablation study"""
    # Validate input files
    if not os.path.exists(args.train):
        logger.error(f"Training file not found: {args.train}")
        return 1

    if not os.path.exists(args.test):
        logger.error(f"Test file not found: {args.test}")
        return 1

    # Set output file
    if not args.output:
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        output_file = os.path.join(EVALUATION_DIR, "ablation_study.json")
    else:
        output_file = args.output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load data
    logger.info(f"Loading training data from {args.train}")
    train_df = pd.read_csv(args.train)

    logger.info(f"Loading test data from {args.test}")
    test_df = pd.read_csv(args.test)

    # Filter test data to only include rows with manual annotations
    test_df = test_df[~pd.isna(test_df['manual_sentiment'])]
    logger.info(f"Found {len(test_df)} samples with manual annotations for testing")

    if len(test_df) == 0:
        logger.error("No manual annotations found in the test file")
        return 1

    # Initialize evaluator with a fresh classifier
    evaluator = EVClassifierEvaluator(EVOpinionClassifier())

    # Run ablation study
    logger.info("Running ablation study...")
    results = evaluator.perform_ablation_study(train_df, test_df, output_path=output_file)

    logger.info(f"Ablation study results saved to {output_file}")
    logger.info(f"Baseline accuracy: {results['baseline']['accuracy']:.4f}")
    logger.info(f"Best model accuracy: {results['combined']['accuracy']:.4f}")
    logger.info(f"Improvement: +{results['combined']['improvement']['accuracy_pct']:.1f}%")

    return 0


def run_train(args):
    """Train classifier models"""
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Set output path
    if not args.output:
        os.makedirs(MODELS_DIR, exist_ok=True)
        output_path = os.path.join(MODELS_DIR, "ev_classifier")
    else:
        output_path = args.output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    logger.info(f"Loading training data from {args.input}")
    df = pd.read_csv(args.input)

    # Initialize classifier
    classifier = EVOpinionClassifier()

    # Build topic model
    logger.info(f"Building topic model with {args.topics} topics...")
    classifier.build_topic_model(df['text'].tolist(), num_topics=args.topics)

    # Save models
    logger.info(f"Saving trained models to {output_path}...")
    classifier.save_models(output_path)

    logger.info("Training complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())