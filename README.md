# Streaming Opinion Search Engine

## Project Overview

This system is an opinion search engine for streaming services, enabling users to find relevant opinions about various streaming platforms (Netflix, Disney+, HBO Max, etc.) and perform sentiment analysis on the results. The system crawls data from social media platforms, indexes it using Apache Solr, and classifies opinions using NLP techniques.

## Features

- **Crawling**: Collects opinions from Reddit and Twitter about streaming services
- **Text Processing**: Cleans, normalizes, and enriches text data
- **Sentiment Analysis**: Classifies opinions as positive, negative, or neutral
- **Feature Extraction**: Analyzes content quality, pricing, UI/UX, technical aspects, and customer service
- **Advanced Search**: Faceted search, filtering by platform, sentiment, and features
- **Visualizations**: Interactive charts for search results analysis
- **Timeline Search**: Search opinions within specific time periods
- **Interactive Feedback**: Refine search results through relevance feedback

## Installation

### Prerequisites

- Python 3.8 or higher
- Apache Solr 9.x
- Git

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/Alaneel/SC4021.git
   cd SC4021
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download NLTK data:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
   ```

5. Download spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```

6. Set up Apache Solr:
   - Download and install Solr 9.x from https://solr.apache.org/downloads.html
   - Start Solr: `bin/solr start` (Unix) or `bin\solr.cmd start` (Windows)
   - Create a core: `bin/solr create -c streaming_opinions` (Unix) or `bin\solr.cmd create -c streaming_opinions` (Windows)
   - Copy schema and configuration:
     ```
     cp solr_files/schema.xml <solr_installation>/server/solr/streaming_opinions/conf/
     ```
   - Restart Solr: `bin/solr restart` (Unix) or `bin\solr.cmd restart` (Windows)

7. Configure credentials:
   - Copy the template: `cp crawler/credentials.py.example crawler/credentials.py`
   - Edit `crawler/credentials.py` with your Reddit and Twitter API credentials

### Running the Crawler

1. Crawl data from Reddit:
   ```
   cd crawler
   python run_crawler.py
   ```

2. Process the crawled data:
   ```
   cd processing
   python processing.py
   ```

3. Import data to Solr:
   ```
   cd indexer
   python import_to_solr.py
   ```

### Starting the Web Application

1. Start the Flask app:
   ```
   python app.py
   ```

2. Access the web interface at http://localhost:5000

## Project Structure

- `app.py`: Main Flask application
- `crawler/`: Data collection modules
  - `crawler.py`: Reddit and Twitter crawlers
  - `config.py`: Crawler configuration
  - `credentials.py`: API credentials (not tracked in Git)
  - `run_crawler.py`: Script to execute crawlers
- `processing/`: Text processing and analysis
  - `processing.py`: Text cleaning, sentiment analysis, and feature extraction
- `indexer/`: Solr integration
  - `import_to_solr.py`: Import processed data to Solr
- `solr_files/`: Solr configuration
  - `schema.xml`: Field definitions for Solr
- `templates/`: HTML templates for web interface
- `evaluation/`: Evaluation framework and datasets
  - `evaluate_classifier.py`: Sentiment classifier evaluation
  - `performance_metrics.py`: Search performance testing
  - `create_evaluation_dataset.py`: Tool for creating labeled datasets
- `advanced_features/`: Additional search capabilities
  - `interactive_feedback.py`: Relevance feedback system
- `tests/`: Test files
  - `integration_test.py`: End-to-end tests

## Creating an Evaluation Dataset

1. Generate a sample for annotation:
   ```
   cd evaluation
   python create_evaluation_dataset.py
   ```

2. Open the annotation tool:
   ```
   python -m evaluation.create_evaluation_dataset ../data/streaming_opinions_dataset.csv
   ```

3. Evaluate classifier performance:
   ```
   python evaluate_classifier.py
   ```

## Performance Testing

Run search performance tests:
```
cd evaluation
python performance_metrics.py
```

## Contributing

See the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was developed as part of the SC4021 course
- Built using Apache Solr, Flask, and various Python NLP libraries
- Test data collected from Reddit and Twitter