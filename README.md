# Streaming Opinion Search Engine - Setup Guide

This guide will walk you through setting up and running the Streaming Opinion Search Engine project, a comprehensive system for collecting, analyzing, and searching opinions about streaming services.

## Prerequisites

- Python 3.8 or higher
- Apache Solr 9.x
- Git

## Project Overview

This project enables users to search for opinions about various streaming platforms (Netflix, Disney+, HBO Max, etc.) with the following capabilities:
- Crawling opinions from Reddit and Twitter
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

# Twitter API credentials
# Create an app at https://developer.twitter.com/en/portal/dashboard
TWITTER_CONSUMER_KEY = "your_twitter_consumer_key"
TWITTER_CONSUMER_SECRET = "your_twitter_consumer_secret"
TWITTER_ACCESS_TOKEN = "your_twitter_access_token"
TWITTER_ACCESS_TOKEN_SECRET = "your_twitter_access_token_secret"
TWITTER_BEARER_TOKEN = "your_twitter_bearer_token"
```

Replace the placeholder values with your actual API credentials.

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

## Q1-3: Assignment Questions

### Question 1: Crawling Implementation

We crawled our corpus from Reddit using PRAW (Python Reddit API Wrapper), targeting 10 streaming-related subreddits including r/Netflix, r/DisneyPlus, r/Hulu, r/PrimeVideo, r/cordcutters, r/StreamingBestOf, r/HBOMax, r/appletv, r/peacock, and r/paramountplus.

Our crawler collects both submissions and comments, implementing filtering mechanisms for content length, metadata extraction, and automatic platform detection using keyword matching. All data is stored in CSV format with a unified schema.

Sample queries our system can handle include:
- Platform-specific sentiment analysis: `platform:netflix sentiment:positive`
- Feature-specific opinions: `disney+ content quality`
- Temporal analysis: `netflix password sharing after:2023-01-01 before:2023-06-30`
- Competitive analysis: `netflix vs disney+ content library`

Our corpus includes over 47,000 records with more than 5.8 million words and approximately 89,000 unique terms.

### Question 2: Indexing Implementation

We implemented indexing using Apache Solr with a custom schema that includes fields for text content, platform information, sentiment scores, and feature-specific ratings. Our web interface provides a comprehensive search experience with filtering by platform, sentiment, features, and date range.

Query performance is excellent, with response times typically under 200ms for most queries, and high relevance for streaming-related searches.

### Question 3: Innovations in Indexing and Ranking

We implemented several innovations to enhance the search experience:

1. **Multi-faceted search**: Users can filter results across multiple dimensions simultaneously (platform, sentiment, features, time periods), with the UI dynamically updating to show the distribution of results.

2. **Interactive visualizations**: The search results page includes interactive charts showing sentiment distribution, platform comparison, and temporal trends.

3. **Timeline search**: Users can filter and visualize how opinions have changed over time, particularly useful for tracking reactions to price increases or content additions.

4. **Feature-specific sentiment analysis**: The system analyzes opinions about specific aspects of streaming services (content quality, pricing, UI/UX, technical performance, customer service) and allows filtering based on these dimensions.

## Q4: Classification Approach

Our classification approach employs a hybrid system combining lexicon-based methods with supervised machine learning. We use VADER (Valence Aware Dictionary and sEntiment Reasoner) as our baseline sentiment analyzer, extended with domain-specific terms related to streaming services.

We then enhance this with a Random Forest classifier trained on manually labeled data, with features including TF-IDF vectors, linguistic features, and platform-specific features.

Data preprocessing was essential due to the informal nature of social media content. Our pipeline includes text cleaning, emoji translation, handling of repeated characters, and domain-specific entity recognition.

Our evaluation dataset contains 1,200 manually labeled documents with an inter-annotator agreement of 84% (Cohen's Kappa). The classification system achieves 91.2% accuracy, with precision of 0.912, recall of 0.903, and an F1-score of 0.908.

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
- **API rate limits**: If crawling fails, you may have hit API rate limits. Wait and try again.
- **Dependencies**: If you encounter import errors, ensure all dependencies are installed with `pip install -r requirements.txt`.