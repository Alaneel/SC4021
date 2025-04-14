# Streaming Opinion Search Engine - Setup Guide

This guide will walk you through setting up and running the Streaming Opinion Search Engine project, a comprehensive system for collecting, analyzing, and searching opinions about streaming services.

## Prerequisites

- Python 3.8 or higher
- Apache Solr 9.x
- Git

## Project Overview

This project enables users to search for opinions about various streaming platforms (Netflix, Disney+, HBO Max, etc.) with the following capabilities:
- Crawling opinions from Reddit
- Comprehensive text processing and sentiment analysis
- Multi-faceted search with filtering by platform, sentiment, and features
- Interactive visualizations of search results
- Aspect-based sentiment analysis for streaming service features

## Setup Process

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/streaming-opinion-search.git
cd streaming-opinion-search
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
```

### 5. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 6. Create Crawler Credentials

Create a file named `credentials.py` in the `crawler` directory with the following content:

```python
# Reddit API credentials
# Create an app at https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID = "your_reddit_client_id"
REDDIT_CLIENT_SECRET = "your_reddit_client_secret"
REDDIT_USER_AGENT = "python:streaming_opinions:v1.0 (by /u/your_username)"
```

Replace the placeholder values with your actual Reddit API credentials.

### 7. Set Up Apache Solr

1. Download and install Solr 9.x from https://solr.apache.org/downloads.html
2. Start Solr: 
   ```bash
   bin/solr start  # On Windows: bin\solr.cmd start
   ```
3. Create a core:
   ```bash
   bin/solr create -c streaming_opinions  # On Windows: bin\solr.cmd create -c streaming_opinions
   ```
4. Copy the schema file:
   ```bash
   cp solr_files/schema.xml [solr_installation_path]/server/solr/streaming_opinions/conf/
   ```
5. Restart Solr:
   ```bash
   bin/solr restart  # On Windows: bin\solr.cmd restart
   ```

## Running the Project

### 1. Crawl Data from Reddit

```bash
cd crawler
python run_crawler.py
cd ..
```

This will collect opinions from streaming-related subreddits and save them to the `data` directory.

### 2. Process the Crawled Data

```bash
cd processing
python processing.py
cd ..
```

This will clean the text, perform sentiment analysis, extract features, and save the processed data.

### 3. Import Data to Solr

```bash
cd indexer
python import_to_solr.py
cd ..
```

This will index the processed data in Solr, making it searchable.

### 4. Start the Web Application

```bash
python app.py
```

Access the web interface at http://localhost:5000

## Creating and Evaluating the Classification System

### 1. Create Evaluation Dataset

Generate a sample dataset for manual annotation:

```bash
cd evaluation
python manual_annotate.py ../data/streaming_opinions_dataset.csv
```

This will open a GUI tool for annotating sentiment and features in the dataset.

### 2. Merge Annotations

After annotation by multiple annotators, merge the results:

```bash
python merge_manual_annotate.py --input-dir data --output-path data/evaluation_dataset_merged.csv
```

### 3. Run Classification

Train and evaluate the sentiment classification model:

```bash
cd classification
python classification.py
```

## Project Structure

- `app.py`: Main Flask application
- `crawler/`: Data collection modules
- `processing/`: Text processing and analysis
- `indexer/`: Solr integration
- `classification/`: Sentiment analysis and feature extraction
- `evaluation/`: Evaluation framework and datasets
- `templates/`: HTML templates for web interface
- `solr_files/`: Solr configuration files

## Troubleshooting

- **Solr connection issues**: Ensure Solr is running (`bin/solr status`) and the core exists.
- **Reddit API rate limits**: If crawling fails, you may have hit API rate limits. Wait and try again.
- **Dependencies**: If you encounter import errors, ensure all dependencies are installed with `pip install -r requirements.txt`