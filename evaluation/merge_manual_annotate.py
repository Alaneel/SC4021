#!/usr/bin/env python3
"""
Script to combine annotations from multiple annotators into a final evaluation dataset.
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

class EvaluationDatasetCreator:
    """Tool for creating and managing evaluation datasets for sentiment classification."""

    def __init__(self, data_path=None, output_path=None, sample_size=1000):
        """
        Initialize the dataset creator.

        Args:
            data_path (str): Path to the processed dataset
            output_path (str): Path to save the evaluation dataset
            sample_size (int): Number of records to include in the evaluation dataset
        """
        self.data_path = data_path
        self.output_path = output_path or 'data/evaluation_dataset.csv'
        self.sample_size = sample_size
        self.data = None
        self.evaluation_sample = None
        self.annotators = {}
        self.current_annotator = None

    def load_data(self, path=None):
        """
        Load the processed dataset.

        Args:
            path (str, optional): Path to the processed dataset

        Returns:
            pd.DataFrame: Loaded dataset
        """
        path = path or self.data_path
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")

        self.data = pd.read_csv(path)
        print(f"Loaded {len(self.data)} records from {path}")
        return self.data

    def create_stratified_sample(self, strata_column='sentiment', strata_ratios=None):
        """
        Create a stratified sample of records for evaluation.

        Args:
            strata_column (str): Column to stratify by
            strata_ratios (dict): Target ratios for each stratum

        Returns:
            pd.DataFrame: Stratified sample
        """
        if self.data is None:
            self.load_data()

        # Get current distribution
        value_counts = self.data[strata_column].value_counts(normalize=True)

        # Use current distribution if no target ratios provided
        if strata_ratios is None:
            strata_ratios = value_counts.to_dict()

        # Calculate samples per stratum
        strata_samples = {}
        remaining_samples = self.sample_size

        for stratum, ratio in strata_ratios.items():
            if stratum == list(strata_ratios.keys())[-1]:
                # Last stratum gets all remaining samples
                strata_samples[stratum] = remaining_samples
            else:
                # Calculate samples for this stratum
                stratum_samples = int(self.sample_size * ratio)
                strata_samples[stratum] = stratum_samples
                remaining_samples -= stratum_samples

        # Create sample dataframe
        sample_dfs = []

        for stratum, count in strata_samples.items():
            # Get records for this stratum
            stratum_data = self.data[self.data[strata_column] == stratum]

            # If we don't have enough records, take all available
            if len(stratum_data) <= count:
                sample_dfs.append(stratum_data)
            else:
                # Otherwise, take a random sample
                sample_dfs.append(stratum_data.sample(n=count, random_state=42))

        # Combine samples
        self.evaluation_sample = pd.concat(sample_dfs, ignore_index=True)

        # Add manual annotation columns
        self.evaluation_sample['manual_sentiment'] = None
        self.evaluation_sample['annotator_id'] = None
        self.evaluation_sample['annotation_timestamp'] = None

        # Add manual feature columns
        feature_columns = [
            'content_quality', 'pricing', 'ui_ux', 'technical', 'customer_service'
        ]

        for feature in feature_columns:
            if feature in self.evaluation_sample.columns:
                self.evaluation_sample[f'manual_{feature}'] = None

        print(f"Created stratified sample with {len(self.evaluation_sample)} records")
        return self.evaluation_sample

    def export_sample_for_annotation(self, output_path=None, annotator_id=None):
        """
        Export the evaluation sample for annotation.

        Args:
            output_path (str, optional): Path to save the sample
            annotator_id (str, optional): ID of the annotator

        Returns:
            str: Path to the exported sample
        """
        if self.evaluation_sample is None:
            self.create_stratified_sample()

        output_path = output_path or f"data/evaluation_sample_{annotator_id or 'default'}.csv"

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Export sample
        self.evaluation_sample.to_csv(output_path, index=False)
        print(f"Exported evaluation sample to {output_path}")
        return output_path

    def import_annotated_sample(self, annotated_path, annotator_id):
        """
        Import an annotated sample and add it to the evaluation dataset.

        Args:
            annotated_path (str): Path to the annotated sample
            annotator_id (str): ID of the annotator

        Returns:
            pd.DataFrame: Updated evaluation sample
        """
        if not os.path.exists(annotated_path):
            raise FileNotFoundError(f"Annotated sample not found at {annotated_path}")

        # Load annotated sample
        annotated_df = pd.read_csv(annotated_path)

        # Add annotator information
        annotated_df['annotator_id'] = annotator_id

        # Store in annotators dictionary
        self.annotators[annotator_id] = annotated_df

        print(f"Imported annotated sample from {annotator_id}")
        return annotated_df

    def merge_annotations(self):
        """
        Merge annotations from multiple annotators.

        Returns:
            pd.DataFrame: Merged evaluation dataset
        """
        if not self.annotators:
            raise ValueError("No annotated samples available")

        # If we only have one annotator, use that data
        if len(self.annotators) == 1:
            annotator_id = list(self.annotators.keys())[0]
            merged_df = self.annotators[annotator_id].copy()
            print(f"Using annotations from {annotator_id}")
            return merged_df

        # For multiple annotators, merge by majority vote
        print(f"Merging annotations from {len(self.annotators)} annotators")

        # Start with the first annotator's data as the base
        base_annotator = list(self.annotators.keys())[0]
        merged_df = self.annotators[base_annotator].copy()

        # Get all record IDs
        all_ids = set()
        for annotator_df in self.annotators.values():
            all_ids.update(annotator_df['id'].values)

        # For each record, get annotations from all annotators
        for record_id in tqdm(all_ids, desc="Merging annotations"):
            # Get annotations for this record
            record_annotations = {}

            for annotator_id, annotator_df in self.annotators.items():
                record_df = annotator_df[annotator_df['id'] == record_id]

                if len(record_df) > 0:
                    # Store annotations for sentiment and features
                    record_annotations[annotator_id] = {
                        'manual_sentiment': record_df['manual_sentiment'].iloc[0]
                    }

                    # Store feature annotations if available
                    for col in record_df.columns:
                        if col.startswith('manual_') and col != 'manual_sentiment':
                            record_annotations[annotator_id][col] = record_df[col].iloc[0]

            # Calculate majority vote for sentiment
            sentiment_votes = {}
            for annotator_data in record_annotations.values():
                sentiment = annotator_data.get('manual_sentiment')
                if sentiment:
                    sentiment_votes[sentiment] = sentiment_votes.get(sentiment, 0) + 1

            if sentiment_votes:
                majority_sentiment = max(sentiment_votes.items(), key=lambda x: x[1])[0]

                # Update merged dataframe
                merged_df.loc[merged_df['id'] == record_id, 'manual_sentiment'] = majority_sentiment

            # Calculate average for feature scores
            feature_cols = [col for col in merged_df.columns if col.startswith('manual_') and col != 'manual_sentiment']

            for feature_col in feature_cols:
                feature_scores = []

                for annotator_data in record_annotations.values():
                    score = annotator_data.get(feature_col)
                    if score and not np.isnan(score):
                        feature_scores.append(score)

                if feature_scores:
                    avg_score = np.mean(feature_scores)
                    merged_df.loc[merged_df['id'] == record_id, feature_col] = avg_score

        print(f"Created merged dataset with {len(merged_df)} records")
        self.evaluation_sample = merged_df
        return merged_df

    def calculate_annotator_agreement(self):
        """
        Calculate agreement metrics between annotators.

        Returns:
            dict: Agreement metrics
        """
        if len(self.annotators) < 2:
            print("At least two annotators required to calculate agreement")
            return {}

        from sklearn.metrics import cohen_kappa_score

        # Calculate agreement for each pair of annotators
        annotator_ids = list(self.annotators.keys())
        agreement_metrics = {}

        for i in range(len(annotator_ids)):
            for j in range(i + 1, len(annotator_ids)):
                annotator1 = annotator_ids[i]
                annotator2 = annotator_ids[j]

                # Get common records
                df1 = self.annotators[annotator1]
                df2 = self.annotators[annotator2]

                common_ids = set(df1['id']) & set(df2['id'])

                if not common_ids:
                    continue

                # Filter to common records
                df1_common = df1[df1['id'].isin(common_ids)]
                df2_common = df2[df2['id'].isin(common_ids)]

                # Sort by ID to ensure alignment
                df1_common = df1_common.sort_values('id')
                df2_common = df2_common.sort_values('id')

                # Calculate agreement for sentiment
                sentiment_agreement = np.mean(df1_common['manual_sentiment'] == df2_common['manual_sentiment'])

                # Calculate Cohen's Kappa for sentiment
                try:
                    sentiment_kappa = cohen_kappa_score(
                        df1_common['manual_sentiment'],
                        df2_common['manual_sentiment']
                    )
                except:
                    sentiment_kappa = None

                # Store metrics
                pair_name = f"{annotator1}_vs_{annotator2}"
                agreement_metrics[pair_name] = {
                    'records': len(common_ids),
                    'sentiment_agreement': sentiment_agreement,
                    'sentiment_kappa': sentiment_kappa,
                }

                # Calculate agreement for features
                feature_cols = [col for col in df1_common.columns
                                if col.startswith('manual_') and col != 'manual_sentiment']

                for feature_col in feature_cols:
                    if feature_col in df1_common.columns and feature_col in df2_common.columns:
                        # Calculate correlation for numeric scores
                        try:
                            correlation = df1_common[feature_col].corr(df2_common[feature_col])
                            agreement_metrics[pair_name][f"{feature_col}_correlation"] = correlation
                        except:
                            pass

        # Calculate average agreement
        if agreement_metrics:
            avg_sentiment_agreement = np.mean([m['sentiment_agreement'] for m in agreement_metrics.values()])
            avg_sentiment_kappa = np.mean([m['sentiment_kappa'] for m in agreement_metrics.values()
                                           if m['sentiment_kappa'] is not None])

            agreement_metrics['overall'] = {
                'avg_sentiment_agreement': avg_sentiment_agreement,
                'avg_sentiment_kappa': avg_sentiment_kappa
            }

        return agreement_metrics

    def save_evaluation_dataset(self, output_path=None):
        """
        Save the final evaluation dataset.

        Args:
            output_path (str, optional): Path to save the dataset

        Returns:
            str: Path to the saved dataset
        """
        output_path = output_path or self.output_path

        if self.evaluation_sample is None:
            raise ValueError("No evaluation sample available")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save dataset
        self.evaluation_sample.to_csv(output_path, index=False)
        print(f"Saved evaluation dataset to {output_path}")
        return output_path

def main():
    """
    Main function to process and combine annotated datasets.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Combine annotated datasets into evaluation dataset')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing annotated CSV files')
    parser.add_argument('--output-path', type=str, default='data/final_evaluation_dataset.csv',
                        help='Path to save the final evaluation dataset')
    parser.add_argument('--agreement-report', type=str, default='data/annotator_agreement_report.json',
                        help='Path to save the annotator agreement report')
    parser.add_argument('--min-agreement', type=float, default=0.7,
                        help='Minimum agreement threshold for including records')

    args = parser.parse_args()

    # Initialize the dataset creator
    creator = EvaluationDatasetCreator(output_path=args.output_path)

    # Find all annotated CSV files in the input directory
    annotated_files = []
    for file in os.listdir(args.input_dir):
        if file.endswith('.csv') and 'annotated' in file:
            annotated_files.append(os.path.join(args.input_dir, file))

    if not annotated_files:
        print(f"No annotated CSV files found in {args.input_dir}")
        return

    print(f"Found {len(annotated_files)} annotated files")

    # Import each annotated file
    for file_path in annotated_files:
        # Extract annotator ID from filename
        filename = os.path.basename(file_path)
        annotator_id = filename.replace('annotated_', '').replace('.csv', '')

        print(f"Importing annotations from {annotator_id}")
        creator.import_annotated_sample(file_path, annotator_id)

    # Calculate annotator agreement
    agreement_metrics = creator.calculate_annotator_agreement()

    if agreement_metrics:
        # Display agreement metrics
        print("\nAnnotator Agreement Metrics:")
        for pair, metrics in agreement_metrics.items():
            if pair == 'overall':
                print(f"\nOverall Agreement:")
                print(f"  Average Sentiment Agreement: {metrics['avg_sentiment_agreement']:.4f}")
                print(f"  Average Sentiment Kappa: {metrics['avg_sentiment_kappa']:.4f}")
            else:
                print(f"\n{pair}:")
                print(f"  Records: {metrics['records']}")
                print(f"  Sentiment Agreement: {metrics['sentiment_agreement']:.4f}")
                if metrics['sentiment_kappa'] is not None:
                    print(f"  Sentiment Kappa: {metrics['sentiment_kappa']:.4f}")

                # Display feature correlations
                for key, value in metrics.items():
                    if key.endswith('_correlation'):
                        feature = key.replace('_correlation', '')
                        print(f"  {feature} Correlation: {value:.4f}")

        # Save agreement report
        import json
        os.makedirs(os.path.dirname(args.agreement_report), exist_ok=True)
        with open(args.agreement_report, 'w') as f:
            json.dump(agreement_metrics, f, indent=2)

        print(f"\nAgreement report saved to {args.agreement_report}")

    # Merge annotations
    print("\nMerging annotations from all annotators...")
    merged_dataset = creator.merge_annotations()

    # Save the final evaluation dataset
    output_path = creator.save_evaluation_dataset()
    print(f"\nFinal evaluation dataset saved to {output_path}")

    # Print dataset statistics
    print("\nEvaluation Dataset Statistics:")
    print(f"  Total records: {len(merged_dataset)}")

    sentiment_counts = merged_dataset['manual_sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} records ({count / len(merged_dataset) * 100:.1f}%)")

    # Check if there are any records without manual sentiment
    missing_sentiment = merged_dataset['manual_sentiment'].isna().sum()
    if missing_sentiment > 0:
        print(f"  Warning: {missing_sentiment} records have missing manual sentiment labels")


if __name__ == "__main__":
    main()