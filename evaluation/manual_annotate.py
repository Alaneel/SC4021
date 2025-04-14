# manual_annotation.py
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


# GUI for annotation
class ManualAnnotator:
    """GUI tool for manually annotating evaluation dataset."""

    def __init__(self, data_path=None):
        """
        Initialize the manual annotator.

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
            initialfile=f"evaluation_dataset_{self.annotator_id}.csv"
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


def manual_annotation_workflow(data_path=None):
    """
    Run the manual annotation workflow.

    Args:
        data_path (str, optional): Path to the dataset to annotate

    Returns:
        None
    """
    # Initialize and run the annotation tool
    annotator = ManualAnnotator(data_path)
    annotator.run()


# Example usage:
if __name__ == "__main__":
    # If run directly, launch the annotation tool
    if len(sys.argv) > 1:
        # Use file path from command line
        manual_annotation_workflow(sys.argv[1])
    else:
        # Start with empty tool
        manual_annotation_workflow()