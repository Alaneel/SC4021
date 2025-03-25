# advanced_features/word_cloud.py
"""Word cloud visualization for streaming opinions."""

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import io
import base64
from PIL import Image
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class OpinionWordCloud:
    """Word cloud visualizer for streaming opinions."""

    def __init__(self, solr_url="http://localhost:8983/solr/streaming_opinions"):
        """
        Initialize the word cloud visualizer.

        Args:
            solr_url (str): URL of the Solr instance
        """
        self.solr_url = solr_url
        self.stop_words = set(stopwords.words('english'))
        # Add domain-specific stopwords
        self.stop_words.update([
            'netflix', 'disney', 'hulu', 'amazon', 'prime',
            'hbo', 'max', 'apple', 'tv', 'streaming', 'service',
            'platform', 'subscription', 'watch', 'watching'
        ])

    def fetch_documents(self, query="*:*", filters=None, field="cleaned_text", rows=500):
        """
        Fetch documents from Solr.

        Args:
            query (str): Search query
            filters (list): List of filter queries
            field (str): Field to use for word cloud
            rows (int): Number of rows to fetch

        Returns:
            list: Document texts
        """
        # Prepare params
        params = {
            'q': query,
            'fl': field,
            'rows': rows,
            'wt': 'json'
        }

        # Add filters if provided
        if filters:
            params['fq'] = filters

        try:
            response = requests.get(f"{self.solr_url}/select", params=params)
            response.raise_for_status()
            results = response.json()

            # Extract text from documents
            docs = results['response']['docs']
            texts = []

            for doc in docs:
                if field in doc and doc[field]:
                    texts.append(doc[field])

            return texts
        except Exception as e:
            print(f"Error fetching documents: {e}")
            return []

    def preprocess_text(self, texts):
        """
        Preprocess text for word cloud.

        Args:
            texts (list): List of document texts

        Returns:
            str: Preprocessed text
        """
        if not texts:
            return ""

        # Join all texts
        all_text = " ".join(texts)

        # Convert to lowercase
        all_text = all_text.lower()

        # Remove punctuation
        all_text = all_text.translate(str.maketrans("", "", string.punctuation))

        # Tokenize
        tokens = word_tokenize(all_text)

        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]

        # Join tokens back to text
        preprocessed_text = " ".join(filtered_tokens)

        return preprocessed_text

    def extract_terms_by_sentiment(self, sentiment, rows=500):
        """
        Extract terms specific to a sentiment.

        Args:
            sentiment (str): Sentiment to extract terms for (positive, negative, neutral)
            rows (int): Number of rows to fetch

        Returns:
            str: Preprocessed text for the sentiment
        """
        # Query for documents with the specified sentiment
        filters = [f"sentiment:{sentiment}"]
        texts = self.fetch_documents(filters=filters, rows=rows)

        # Preprocess the texts
        return self.preprocess_text(texts)

    def extract_terms_by_feature(self, feature, threshold=0.5, rows=500):
        """
        Extract terms specific to a feature.

        Args:
            feature (str): Feature to extract terms for (content_quality, pricing, etc.)
            threshold (float): Minimum feature score
            rows (int): Number of rows to fetch

        Returns:
            str: Preprocessed text for the feature
        """
        # Query for documents with high scores for the feature
        filters = [f"{feature}:[{threshold} TO *]"]
        texts = self.fetch_documents(filters=filters, rows=rows)

        # Preprocess the texts
        return self.preprocess_text(texts)

    def extract_terms_by_platform(self, platform, rows=500):
        """
        Extract terms specific to a platform.

        Args:
            platform (str): Platform to extract terms for (netflix, disney+, etc.)
            rows (int): Number of rows to fetch

        Returns:
            str: Preprocessed text for the platform
        """
        # Query for documents for the specified platform
        filters = [f"platform:{platform}"]
        texts = self.fetch_documents(filters=filters, rows=rows)

        # Preprocess the texts
        return self.preprocess_text(texts)

    def extract_comparative_terms(self, entity1, entity2, field="platform", rows=300):
        """
        Extract terms that compare two entities.

        Args:
            entity1 (str): First entity to compare
            entity2 (str): Second entity to compare
            field (str): Field to use for comparison (platform, sentiment, etc.)
            rows (int): Number of rows to fetch per entity

        Returns:
            tuple: (entity1_text, entity2_text, common_terms)
        """
        # Query for documents for entity1
        filters1 = [f"{field}:{entity1}"]
        texts1 = self.fetch_documents(filters=filters1, rows=rows)

        # Query for documents for entity2
        filters2 = [f"{field}:{entity2}"]
        texts2 = self.fetch_documents(filters=filters2, rows=rows)

        # Preprocess the texts
        preprocessed1 = self.preprocess_text(texts1)
        preprocessed2 = self.preprocess_text(texts2)

        # Find common terms
        tokens1 = set(word_tokenize(preprocessed1))
        tokens2 = set(word_tokenize(preprocessed2))
        common_tokens = tokens1.intersection(tokens2)

        # Join common tokens
        common_text = " ".join(common_tokens)

        return preprocessed1, preprocessed2, common_text

    def create_word_cloud(self, text, title="Word Cloud", width=800, height=400,
                          background_color="white", colormap="viridis", max_words=200):
        """
        Create a word cloud visualization.

        Args:
            text (str): Text for word cloud
            title (str): Chart title
            width (int): Width of the visualization
            height (int): Height of the visualization
            background_color (str): Background color
            colormap (str): Colormap for words
            max_words (int): Maximum number of words to include

        Returns:
            tuple: (plt.Figure, base64 encoded image)
        """
        if not text:
            # Create empty figure
            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=16)
            ax.axis('off')

            # Convert to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            return fig, img_base64

        # Create word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            colormap=colormap,
            max_words=max_words,
            collocations=False,
            stopwords=self.stop_words
        ).generate(text)

        # Create figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=16)
        ax.axis('off')

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return fig, img_base64

    def create_comparative_word_cloud(self, entity1, entity2, field="platform",
                                      title="Comparative Word Cloud", width=1000, height=600):
        """
        Create a comparative word cloud visualization.

        Args:
            entity1 (str): First entity to compare
            entity2 (str): Second entity to compare
            field (str): Field to use for comparison (platform, sentiment, etc.)
            title (str): Chart title
            width (int): Width of the visualization
            height (int): Height of the visualization

        Returns:
            tuple: (plt.Figure, base64 encoded image)
        """
        # Get comparative terms
        text1, text2, common_text = self.extract_comparative_terms(entity1, entity2, field)

        if not text1 or not text2:
            # Create empty figure
            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=16)
            ax.axis('off')

            # Convert to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            return fig, img_base64

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width / 100, height / 100), dpi=100)

        # Create word cloud for entity1
        wordcloud1 = WordCloud(
            width=int(width / 2),
            height=height,
            background_color="white",
            colormap="Blues",
            max_words=100,
            collocations=False,
            stopwords=self.stop_words
        ).generate(text1)

        # Create word cloud for entity2
        wordcloud2 = WordCloud(
            width=int(width / 2),
            height=height,
            background_color="white",
            colormap="Reds",
            max_words=100,
            collocations=False,
            stopwords=self.stop_words
        ).generate(text2)

        # Show word clouds
        ax1.imshow(wordcloud1, interpolation='bilinear')
        ax1.set_title(f"{entity1} Terms", fontsize=16)
        ax1.axis('off')

        ax2.imshow(wordcloud2, interpolation='bilinear')
        ax2.set_title(f"{entity2} Terms", fontsize=16)
        ax2.axis('off')

        fig.suptitle(title, fontsize=20)

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return fig, img_base64

    def create_sentiment_word_clouds(self, width=1000, height=1000):
        """
        Create sentiment-specific word clouds.

        Args:
            width (int): Width of the visualization
            height (int): Height of the visualization

        Returns:
            tuple: (plt.Figure, base64 encoded image)
        """
        # Extract terms for each sentiment
        positive_text = self.extract_terms_by_sentiment("positive")
        negative_text = self.extract_terms_by_sentiment("negative")
        neutral_text = self.extract_terms_by_sentiment("neutral")

        if not positive_text and not negative_text and not neutral_text:
            # Create empty figure
            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=16)
            ax.axis('off')

            # Convert to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            return fig, img_base64

        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(width / 100, height / 100), dpi=100)

        # Create word cloud for positive sentiment
        if positive_text:
            wordcloud1 = WordCloud(
                width=width,
                height=int(height / 3),
                background_color="white",
                colormap="Greens",
                max_words=100,
                collocations=False,
                stopwords=self.stop_words
            ).generate(positive_text)

            ax1.imshow(wordcloud1, interpolation='bilinear')
            ax1.set_title("Positive Sentiment Terms", fontsize=16)
            ax1.axis('off')
        else:
            ax1.text(0.5, 0.5, "No positive sentiment data", ha='center', va='center', fontsize=14)
            ax1.axis('off')

        # Create word cloud for negative sentiment
        if negative_text:
            wordcloud2 = WordCloud(
                width=width,
                height=int(height / 3),
                background_color="white",
                colormap="Reds",
                max_words=100,
                collocations=False,
                stopwords=self.stop_words
            ).generate(negative_text)

            ax2.imshow(wordcloud2, interpolation='bilinear')
            ax2.set_title("Negative Sentiment Terms", fontsize=16)
            ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, "No negative sentiment data", ha='center', va='center', fontsize=14)
            ax2.axis('off')

        # Create word cloud for neutral sentiment
        if neutral_text:
            wordcloud3 = WordCloud(
                width=width,
                height=int(height / 3),
                background_color="white",
                colormap="Greys",
                max_words=100,
                collocations=False,
                stopwords=self.stop_words
            ).generate(neutral_text)

            ax3.imshow(wordcloud3, interpolation='bilinear')
            ax3.set_title("Neutral Sentiment Terms", fontsize=16)
            ax3.axis('off')
        else:
            ax3.text(0.5, 0.5, "No neutral sentiment data", ha='center', va='center', fontsize=14)
            ax3.axis('off')

        fig.suptitle("Sentiment-Specific Word Clouds", fontsize=20)
        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return fig, img_base64

    def create_feature_word_clouds(self, features=None, width=1000, height=800):
        """
        Create feature-specific word clouds.

        Args:
            features (list): List of features to visualize
            width (int): Width of the visualization
            height (int): Height of the visualization

        Returns:
            tuple: (plt.Figure, base64 encoded image)
        """
        # Default features if not provided
        if features is None:
            features = [
                'content_quality', 'pricing', 'ui_ux', 'technical', 'customer_service'
            ]

        # Feature display names
        feature_names = {
            'content_quality': 'Content Quality',
            'pricing': 'Pricing',
            'ui_ux': 'User Interface',
            'technical': 'Technical Performance',
            'customer_service': 'Customer Service'
        }

        # Feature colormaps
        feature_colors = {
            'content_quality': 'Greens',
            'pricing': 'Blues',
            'ui_ux': 'Purples',
            'technical': 'Oranges',
            'customer_service': 'Reds'
        }

        # Extract terms for each feature
        feature_texts = {}
        for feature in features:
            feature_texts[feature] = self.extract_terms_by_feature(feature)

        if not any(feature_texts.values()):
            # Create empty figure
            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=16)
            ax.axis('off')

            # Convert to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            return fig, img_base64

        # Determine grid layout
        num_features = len(features)
        if num_features <= 3:
            nrows, ncols = 1, num_features
        elif num_features <= 6:
            nrows, ncols = 2, (num_features + 1) // 2
        else:
            nrows, ncols = 3, (num_features + 2) // 3

        # Create figure with subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=(width / 100, height / 100), dpi=100)

        # Ensure axes is a 2D array
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            axes = np.array([[ax] for ax in axes])

        # Create word cloud for each feature
        for i, feature in enumerate(features):
            row, col = i // ncols, i % ncols
            ax = axes[row, col]

            if feature_texts[feature]:
                wordcloud = WordCloud(
                    width=int(width / ncols),
                    height=int(height / nrows),
                    background_color="white",
                    colormap=feature_colors.get(feature, "viridis"),
                    max_words=100,
                    collocations=False,
                    stopwords=self.stop_words
                ).generate(feature_texts[feature])

                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(feature_names.get(feature, feature), fontsize=14)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, f"No {feature} data", ha='center', va='center', fontsize=12)
                ax.axis('off')

        # Hide any unused subplots
        for i in range(len(features), nrows * ncols):
            row, col = i // ncols, i % ncols
            axes[row, col].axis('off')

        fig.suptitle("Feature-Specific Word Clouds", fontsize=20)
        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return fig, img_base64

    def get_word_frequencies(self, text, top_n=30):
        """
        Get word frequencies from text.

        Args:
            text (str): Text to analyze
            top_n (int): Number of top words to return

        Returns:
            list: List of (word, frequency) tuples
        """
        if not text:
            return []

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]

        # Count word frequencies
        word_freq = Counter(filtered_tokens)

        # Get top N words
        top_words = word_freq.most_common(top_n)

        return top_words

    def create_frequency_analysis(self, query="*:*", filters=None, top_n=30,
                                  title="Word Frequency Analysis", width=800, height=600):
        """
        Create a word frequency analysis visualization.

        Args:
            query (str): Search query
            filters (list): List of filter queries
            top_n (int): Number of top words to display
            title (str): Chart title
            width (int): Width of the visualization
            height (int): Height of the visualization

        Returns:
            tuple: (plt.Figure, base64 encoded image)
        """
        # Fetch documents
        texts = self.fetch_documents(query=query, filters=filters)

        if not texts:
            # Create empty figure
            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=16)
            ax.axis('off')

            # Convert to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            return fig, img_base64

        # Preprocess text
        preprocessed_text = self.preprocess_text(texts)

        # Get word frequencies
        word_freqs = self.get_word_frequencies(preprocessed_text, top_n=top_n)

        if not word_freqs:
            # Create empty figure
            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
            ax.text(0.5, 0.5, "No significant words found", ha='center', va='center', fontsize=16)
            ax.axis('off')

            # Convert to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            return fig, img_base64

        # Create figure
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

        # Extract words and frequencies
        words = [word for word, freq in word_freqs]
        freqs = [freq for word, freq in word_freqs]

        # Create horizontal bar chart
        y_pos = np.arange(len(words))
        bars = ax.barh(y_pos, freqs, align='center')

        # Add word labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)

        # Add frequency labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                    str(freqs[i]), ha='left', va='center')

        # Set chart title and labels
        ax.set_title(title)
        ax.set_xlabel('Frequency')

        # Reverse y-axis to show highest frequency at the top
        ax.invert_yaxis()

        plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return fig, img_base64


# Example Flask integration
def integrate_with_flask(app, solr_url):
    """
    Integrate word cloud visualizer with Flask app.

    Args:
        app: Flask application
        solr_url (str): URL of the Solr instance
    """
    from flask import request, jsonify, render_template

    word_cloud = OpinionWordCloud(solr_url)

    @app.route('/word_cloud')
    def word_cloud_page():
        """Render word cloud page."""
        return render_template('word_cloud.html')

    @app.route('/api/word_cloud')
    def generate_word_cloud():
        """API endpoint for generating word cloud."""
        query = request.args.get('q', '*:*')
        filters = request.args.getlist('fq')

        # Fetch documents
        texts = word_cloud.fetch_documents(query=query, filters=filters)

        # Preprocess text
        preprocessed_text = word_cloud.preprocess_text(texts)

        # Create word cloud
        _, img_base64 = word_cloud.create_word_cloud(
            preprocessed_text,
            title=f"Word Cloud: {query}"
        )

        return jsonify({
            'image': f"data:image/png;base64,{img_base64}"
        })

    @app.route('/api/sentiment_word_clouds')
    def generate_sentiment_word_clouds():
        """API endpoint for generating sentiment word clouds."""
        # Create sentiment word clouds
        _, img_base64 = word_cloud.create_sentiment_word_clouds()

        return jsonify({
            'image': f"data:image/png;base64,{img_base64}"
        })

    @app.route('/api/feature_word_clouds')
    def generate_feature_word_clouds():
        """API endpoint for generating feature word clouds."""
        # Create feature word clouds
        _, img_base64 = word_cloud.create_feature_word_clouds()

        return jsonify({
            'image': f"data:image/png;base64,{img_base64}"
        })

    @app.route('/api/comparative_word_cloud')
    def generate_comparative_word_cloud():
        """API endpoint for generating comparative word cloud."""
        entity1 = request.args.get('entity1', 'netflix')
        entity2 = request.args.get('entity2', 'disney+')
        field = request.args.get('field', 'platform')

        # Create comparative word cloud
        _, img_base64 = word_cloud.create_comparative_word_cloud(
            entity1, entity2, field,
            title=f"Comparative Word Cloud: {entity1} vs {entity2}"
        )

        return jsonify({
            'image': f"data:image/png;base64,{img_base64}"
        })

    @app.route('/api/word_frequencies')
    def generate_word_frequencies():
        """API endpoint for generating word frequency analysis."""
        query = request.args.get('q', '*:*')
        filters = request.args.getlist('fq')

        # Create frequency analysis
        _, img_base64 = word_cloud.create_frequency_analysis(
            query=query,
            filters=filters,
            title=f"Word Frequency Analysis: {query}"
        )

        return jsonify({
            'image': f"data:image/png;base64,{img_base64}"
        })