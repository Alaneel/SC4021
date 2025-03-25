# evaluation/create_evaluation_dataset.py
"""Tool for creating labeled evaluation dataset for sentiment classifier."""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


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


# GUI for annotation
class AnnotationTool:
    """GUI tool for manually annotating evaluation dataset."""

    def __init__(self, data_path=None):
        """
        Initialize the annotation tool.

        Args:
            data_path (str, optional): Path to dataset to annotate
        """
        self.data_path = data_path
        self.data = None
        self.current_index = 0
        self.annotator_id = "annotator_1"

        if data_path:
            self.load_data(data_path)

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Opinion Annotation Tool")
        self.root.geometry("1000x800")

        # Create UI layout
        self.create_widgets()

    def load_data(self, path):
        """
        Load dataset for annotation.

        Args:
            path (str): Path to dataset
        """
        try:
            self.data = pd.read_csv(path)
            print(f"Loaded {len(self.data)} records from {path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = None

    def create_widgets(self):
        """Create the UI widgets."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # File controls
        file_frame = ttk.LabelFrame(main_frame, text="File Controls", padding=10)
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(file_frame, text="Load Data", command=self.open_file_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Save Annotations", command=self.save_annotations).pack(side=tk.LEFT, padx=5)

        self.annotator_entry = ttk.Entry(file_frame, width=20)
        self.annotator_entry.insert(0, self.annotator_id)
        self.annotator_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(file_frame, text="Annotator ID").pack(side=tk.LEFT)

        self.progress_label = ttk.Label(file_frame, text="Progress: 0/0")
        self.progress_label.pack(side=tk.RIGHT, padx=5)

        # Content frame
        content_frame = ttk.LabelFrame(main_frame, text="Post Content", padding=10)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Post metadata
        meta_frame = ttk.Frame(content_frame)
        meta_frame.pack(fill=tk.X, pady=5)

        ttk.Label(meta_frame, text="ID:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.id_label = ttk.Label(meta_frame, text="")
        self.id_label.grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(meta_frame, text="Platform:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.platform_label = ttk.Label(meta_frame, text="")
        self.platform_label.grid(row=0, column=3, sticky=tk.W, padx=5)

        ttk.Label(meta_frame, text="Type:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.type_label = ttk.Label(meta_frame, text="")
        self.type_label.grid(row=0, column=5, sticky=tk.W, padx=5)

        ttk.Label(meta_frame, text="Date:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.date_label = ttk.Label(meta_frame, text="")
        self.date_label.grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(meta_frame, text="Original Sentiment:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.orig_sentiment_label = ttk.Label(meta_frame, text="")
        self.orig_sentiment_label.grid(row=1, column=3, sticky=tk.W, padx=5)

        # Text content
        ttk.Label(content_frame, text="Title:").pack(anchor=tk.W)
        self.title_text = tk.Text(content_frame, height=2, wrap=tk.WORD)
        self.title_text.pack(fill=tk.X, pady=5)
        self.title_text.config(state=tk.DISABLED)

        ttk.Label(content_frame, text="Content:").pack(anchor=tk.W)
        self.content_text = tk.Text(content_frame, height=10, wrap=tk.WORD)
        self.content_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.content_text.config(state=tk.DISABLED)

        # Annotation controls
        annotation_frame = ttk.LabelFrame(main_frame, text="Annotation", padding=10)
        annotation_frame.pack(fill=tk.X, padx=5, pady=5)

        # Sentiment selection
        sentiment_frame = ttk.Frame(annotation_frame)
        sentiment_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sentiment_frame, text="Sentiment:").pack(side=tk.LEFT, padx=5)

        self.sentiment_var = tk.StringVar()
        ttk.Radiobutton(sentiment_frame, text="Positive", variable=self.sentiment_var, value="positive").pack(
            side=tk.LEFT, padx=10)
        ttk.Radiobutton(sentiment_frame, text="Negative", variable=self.sentiment_var, value="negative").pack(
            side=tk.LEFT, padx=10)
        ttk.Radiobutton(sentiment_frame, text="Neutral", variable=self.sentiment_var, value="neutral").pack(
            side=tk.LEFT, padx=10)

        # Feature sliders
        features_frame = ttk.Frame(annotation_frame)
        features_frame.pack(fill=tk.X, pady=10)

        self.feature_sliders = {}
        feature_names = [
            ("content_quality", "Content Quality"),
            ("pricing", "Pricing"),
            ("ui_ux", "UI/UX"),
            ("technical", "Technical"),
            ("customer_service", "Customer Service")
        ]

        for i, (feature_id, feature_label) in enumerate(feature_names):
            ttk.Label(features_frame, text=feature_label + ":").grid(row=i, column=0, sticky=tk.W, padx=5, pady=3)

            var = tk.DoubleVar()
            self.feature_sliders[feature_id] = var

            slider = ttk.Scale(features_frame, from_=-1.0, to=1.0, orient=tk.HORIZONTAL,
                               variable=var, length=300)
            slider.grid(row=i, column=1, sticky=tk.W, padx=5, pady=3)

            value_label = ttk.Label(features_frame, textvariable=var)
            value_label.grid(row=i, column=2, sticky=tk.W, padx=5, pady=3)

        # Navigation buttons
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(nav_frame, text="Previous", command=self.prev_record).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_record).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Save Current", command=self.save_current).pack(side=tk.LEFT, padx=5)

    def open_file_dialog(self):
        """Open file dialog to select dataset."""
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filepath:
            self.data_path = filepath
            self.load_data(filepath)
            self.current_index = 0
            self.update_display()

    def save_annotations(self):
        """Save annotations to file."""
        if self.data is None:
            messagebox.showwarning("Warning", "No data to save!")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"annotated_{self.annotator_id}.csv"
        )

        if filepath:
            # Update annotator ID
            self.annotator_id = self.annotator_entry.get()

            # Add annotation timestamp
            self.data['annotation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Save to file
            self.data.to_csv(filepath, index=False)
            messagebox.showinfo("Success", f"Annotations saved to {filepath}")

    def update_display(self):
        """Update the display with the current record."""
        if self.data is None or len(self.data) == 0:
            messagebox.showinfo("Info", "No data to display!")
            return

        # Update progress label
        self.progress_label.config(text=f"Progress: {self.current_index + 1}/{len(self.data)}")

        # Get current record
        record = self.data.iloc[self.current_index]

        # Update metadata labels
        self.id_label.config(text=str(record.get('id', '')))
        self.platform_label.config(text=str(record.get('platform', '')))
        self.type_label.config(text=str(record.get('type', '')))
        self.date_label.config(text=str(record.get('created_at', '')))
        self.orig_sentiment_label.config(text=str(record.get('sentiment', '')))

        # Update text content
        self.title_text.config(state=tk.NORMAL)
        self.title_text.delete('1.0', tk.END)
        if 'title' in record and not pd.isna(record['title']):
            self.title_text.insert(tk.END, str(record['title']))
        self.title_text.config(state=tk.DISABLED)

        self.content_text.config(state=tk.NORMAL)
        self.content_text.delete('1.0', tk.END)
        if 'text' in record and not pd.isna(record['text']):
            self.content_text.insert(tk.END, str(record['text']))
        self.content_text.config(state=tk.DISABLED)

        # Set sentiment selection
        if 'manual_sentiment' in record and not pd.isna(record['manual_sentiment']):
            self.sentiment_var.set(str(record['manual_sentiment']))
        else:
            self.sentiment_var.set('')

        # Set feature sliders
        for feature_id, slider_var in self.feature_sliders.items():
            manual_col = f'manual_{feature_id}'
            if manual_col in record and not pd.isna(record[manual_col]):
                slider_var.set(float(record[manual_col]))
            else:
                slider_var.set(0.0)

    def save_current(self):
        """Save annotations for the current record."""
        if self.data is None or len(self.data) == 0:
            return

        # Get sentiment selection
        sentiment = self.sentiment_var.get()
        if sentiment:
            self.data.at[self.current_index, 'manual_sentiment'] = sentiment

        # Get feature slider values
        for feature_id, slider_var in self.feature_sliders.items():
            manual_col = f'manual_{feature_id}'
            if manual_col in self.data.columns:
                self.data.at[self.current_index, manual_col] = slider_var.get()

        # Save annotator ID
        self.annotator_id = self.annotator_entry.get()
        self.data.at[self.current_index, 'annotator_id'] = self.annotator_id

        # Save timestamp
        self.data.at[self.current_index, 'annotation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        messagebox.showinfo("Success", "Annotations saved for current record")

    def prev_record(self):
        """Navigate to the previous record."""
        if self.data is None or len(self.data) == 0:
            return

        # Save current record
        self.save_current()

        # Go to previous record
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def next_record(self):
        """Navigate to the next record."""
        if self.data is None or len(self.data) == 0:
            return

        # Save current record
        self.save_current()

        # Go to next record
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.update_display()

    def run(self):
        """Run the annotation tool."""
        # Update display if we have data
        if self.data is not None:
            self.update_display()

        # Start the main loop
        self.root.mainloop()


# Example usage:
if __name__ == "__main__":
    # If run directly, launch the annotation tool
    if len(sys.argv) > 1:
        # Use file path from command line
        tool = AnnotationTool(sys.argv[1])
    else:
        # Start with empty tool
        tool = AnnotationTool()

    tool.run()