# EV Opinion Search Engine

A comprehensive search engine for analyzing and exploring opinions about electric vehicles from news sources worldwide.

## Overview

The EV Opinion Search Engine is a specialized tool for collecting, analyzing, and searching public opinions about electric vehicles. It combines web crawling, information retrieval, sentiment analysis, and topic modeling to provide a complete solution for understanding consumer perceptions of EVs.

Key features:
- Crawls and collects EV-related articles from news sources via News API
- Indexes content for fast and efficient searching
- Analyzes sentiment (positive, negative, neutral)
- Identifies topics and entities mentioned in opinions
- Provides a user-friendly web interface with data visualizations

## Project Structure

```
SC4021/
│
├── config/               # Configuration files
│   ├── app_config.py     # Application settings
│   └── solr_schema.xml   # Solr schema definition
│
├── crawler/              # Data collection module
│   ├── newsapi_crawler.py # News API data collector
│   └── data_cleaner.py   # Text cleaning utilities
│
├── indexing/             # Search indexing module
│   ├── solr_indexer.py   # Solr indexing functionality
│   └── search_utils.py   # Search utility functions
│
├── classification/       # Opinion analysis module
│   ├── classifier.py     # Sentiment and topic analysis
│   └── evaluation.py     # Evaluation and annotation tools
│
├── web/                  # Web interface
│   ├── app.py            # Flask application
│   ├── templates/        # HTML templates
│   └── static/           # Static assets (CSS, JS)
│
├── scripts/              # Command-line scripts
│   ├── run_crawler.py    # Run the crawler
│   ├── run_indexer.py    # Run the indexer
│   ├── run_evaluation.py # Run evaluation tools
│   └── run_webapp.py     # Run the web application
│
├── data/                 # Data directory (not in version control)
│   ├── raw/              # Raw crawled data
│   ├── processed/        # Processed data
│   └── evaluation/       # Evaluation datasets
│
└── requirements.txt      # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Apache Solr 8.11 or higher
- News API key (free tier available)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/alaneel/SC4021.git
   cd SC4021
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required models:
   ```bash
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt
   ```

5. Set up environment variables:
   ```bash
   # News API credentials (required for crawler)
   export NEWSAPI_API_KEY="your_api_key"
   
   # Solr configuration (optional, defaults provided)
   export SOLR_URL="http://localhost:8983/solr"
   export SOLR_COLLECTION="ev_opinions"
   ```

6. Set up Solr:
   - Download and install Apache Solr from https://solr.apache.org/
   - Create a collection using the schema provided in `config/solr_schema.xml`

## Usage

### Data Collection

To crawl EV opinions from news sources:

```bash
python scripts/run_crawler.py --limit 100 --preprocess
```

Options:
- `--queries`: Specify search queries (default: from config)
- `--limit`: Maximum articles per query (default: 100)
- `--days`: Number of days to look back (default: 30)
- `--language`: Language of articles (default: en)
- `--preprocess`: Apply text preprocessing to clean the data
- `--output`: Specify output file name
- `--api-key`: Specify News API Key (or use environment variable)

### Indexing

To index the collected data to Solr:

```bash
python scripts/run_indexer.py --latest --classify
```

Options:
- `--input`: Specify input CSV file
- `--latest`: Use the most recent data file
- `--classify`: Run sentiment analysis and topic modeling before indexing
- `--clear`: Clear existing index before indexing
- `--optimize`: Optimize the index after indexing

### Classification and Evaluation

To create an annotation dataset:

```bash
python scripts/run_evaluation.py annotate --latest --samples 1000
```

To evaluate the classifier on annotated data:

```bash
python scripts/run_evaluation.py evaluate --input data/evaluation/ev_opinions_annotation.csv
```

To train the classifier models:

```bash
python scripts/run_evaluation.py train --input data/evaluation/ev_opinions_annotation.csv
```

### Web Application

To run the web interface:

```bash
python scripts/run_webapp.py
```

Then open your browser and navigate to `http://localhost:5000`.

## Classification Approach

The classification module implements a hybrid approach combining:

1. **Sentiment Analysis**: Uses a fine-tuned DistilBERT model to classify opinions as positive, negative, or neutral
2. **Topic Modeling**: Uses Latent Dirichlet Allocation (LDA) to discover hidden topics in the corpus
3. **Entity Recognition**: Identifies EV-related entities such as brands, models, and components

## Web Interface Features

The web interface provides:

- Full-text search with faceted navigation
- Sentiment distribution visualization
- Opinion timeline charts
- Interactive word clouds
- Filtering by sentiment, date, source, topics, and entities

## News API Usage and Limits

When using the News API for data collection, be aware of the following:

1. **Rate Limits**: News API free tier allows 100 requests per day with up to 100 results per request, which is sufficient for collecting around 10,000 articles daily.

2. **Data Timeframe**: Free tier access limits searches to articles published in the last month only.

3. **Search Limitations**: Some filtering capabilities (like by source domain or language) may be limited compared to the paid tier.

4. **Attribution Requirements**: When displaying news content, proper attribution to the source is required according to News API terms of service.

The crawler includes automatic rate limit handling and will save intermediate results to prevent data loss if limits are reached.

## Development

### Adding New Data Sources

To add a new data source, create a new crawler in the `crawler` module following the pattern established by `newsapi_crawler.py`.

### Extending Classification

To add new classification capabilities:
1. Extend the `EVOpinionClassifier` class in `classification/classifier.py`
2. Add evaluation methods in `classification/evaluation.py`
3. Update the indexing script to include the new attributes

## License

[MIT License](LICENSE)

## Acknowledgements

- This project uses the Transformers library by Hugging Face for sentiment analysis
- Topic modeling is implemented using Gensim
- Web interface built with Flask and Bootstrap
- News API for data collection