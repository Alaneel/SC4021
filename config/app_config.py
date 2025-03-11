"""
Application configuration parameters for EV Opinion Search Engine
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EVALUATION_DIR = os.path.join(DATA_DIR, 'evaluation')
MODELS_DIR = os.path.join(BASE_DIR, 'classification', 'models')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EVALUATION_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# X (Twitter) API Configuration
X_API_KEY = os.environ.get('X_API_KEY', '')
X_API_SECRET = os.environ.get('X_API_SECRET', '')
X_BEARER_TOKEN = os.environ.get('X_BEARER_TOKEN', '')
X_ACCESS_TOKEN = os.environ.get('X_ACCESS_TOKEN', '')
X_ACCESS_SECRET = os.environ.get('X_ACCESS_SECRET', '')
X_USER_AGENT = os.environ.get('X_USER_AGENT', 'EV Opinion Search Engine v1.0')

# Search queries for X
SEARCH_QUERIES = [
    "electric vehicle", "EV", "Tesla", "Rivian", "Lucid Motors", "Bolt EV",
    "Nissan Leaf", "EV charging", "electric car", "battery range",
    "EV cost", "EV infrastructure", "NIO", "Polestar", "Hyundai Ioniq",
    "Kia EV6", "GM electric", "Ford Mach-E", "BEV", "PHEV"
]

# Solr configuration
SOLR_URL = os.environ.get('SOLR_URL', 'http://localhost:8983/solr')
SOLR_COLLECTION = os.environ.get('SOLR_COLLECTION', 'ev_opinions')
SOLR_FULL_URL = f"{SOLR_URL}/{SOLR_COLLECTION}"

# Web application configuration
FLASK_SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'dev_key_change_in_production')
FLASK_HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.environ.get('FLASK_PORT', 5000))
FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')

# Classification configuration
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
NUM_TOPICS = 15
SENTIMENT_THRESHOLD = 0.6  # Threshold for positive/negative classification