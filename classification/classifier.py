"""
Main classification implementation for EV opinion analysis
"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import logging as transformers_logging
import spacy
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pickle
import os
import re
import logging
import time
from tqdm import tqdm
import sys

from config.app_config import SENTIMENT_MODEL, NUM_TOPICS, MODELS_DIR, SENTIMENT_THRESHOLD

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classifier.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Reduce verbosity of transformers warnings
transformers_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


class EVOpinionClassifier:
    """
    Classifier for EV opinions, combining sentiment analysis and topic modeling
    """

    def __init__(self, sentiment_model=None, use_gpu=None):
        """
        Initialize the classifier

        Args:
            sentiment_model (str): Pre-trained model name for sentiment analysis
            use_gpu (bool): Whether to use GPU for inference if available
        """
        self.sentiment_model_name = sentiment_model or SENTIMENT_MODEL
        self.sentiment_model = None
        self.sentiment_tokenizer = None
        self.sentiment_pipeline = None

        self.topic_model = None
        self.dictionary = None
        self.corpus = None

        self.nlp = None

        # Determine device for inference
        if use_gpu is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        logger.info(f"Using device: {self.device}")

        # Initialize spaCy
        logger.info("Initializing spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            logger.error("You may need to download it with: python -m spacy download en_core_web_sm")

    def load_sentiment_model(self):
        """
        Load the pre-trained sentiment analysis model
        """
        if self.sentiment_pipeline is not None:
            return  # Already loaded

        logger.info(f"Loading sentiment model: {self.sentiment_model_name}")

        try:
            # Load tokenizer and model
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                self.sentiment_model_name
            ).to(self.device)

            # Create pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.sentiment_model,
                tokenizer=self.sentiment_tokenizer,
                device=0 if self.device.type == 'cuda' else -1
            )

            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            raise

    def preprocess_text(self, text):
        """
        Preprocess text for analysis

        Args:
            text (str): The input text

        Returns:
            str: Preprocessed text
        """
        if not text or pd.isna(text):
            return ""

        # Convert to string if needed
        if not isinstance(text, str):
            text = str(text)

        # Basic cleaning
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?;:()\-\']', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Use spaCy for more advanced preprocessing (if it's available)
        if self.nlp is not None:
            try:
                doc = self.nlp(text)

                # Remove stopwords and lemmatize (for topic modeling)
                tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
                return " ".join(tokens)
            except Exception as e:
                logger.warning(f"Error in spaCy preprocessing: {str(e)}")
                # Fallback to basic preprocessing
                return text
        else:
            return text

    def extract_entities(self, text):
        """
        Extract named entities and EV-related entities from text

        Args:
            text (str): Input text

        Returns:
            list: Extracted entities
        """
        if not text or pd.isna(text):
            return []

        entities = []

        # Use spaCy for named entity recognition
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'PERSON']:
                        entities.append(ent.text.lower())
            except Exception as e:
                logger.warning(f"Error in entity extraction: {str(e)}")

        # Also check for EV-specific entities that might not be recognized
        ev_entities = {
            "tesla": ["tesla", "model s", "model 3", "model y", "model x", "cybertruck"],
            "rivian": ["rivian", "r1t", "r1s"],
            "lucid": ["lucid", "air"],
            "gm": ["chevrolet bolt", "bolt ev", "bolt euv", "chevy bolt", "gm"],
            "ford": ["ford", "mustang mach e", "mach e", "f-150 lightning"],
            "nissan": ["nissan leaf", "leaf", "ariya"],
            "volkswagen": ["volkswagen", "vw", "id.4", "id4"],
            "hyundai": ["hyundai", "ioniq", "kona"],
            "kia": ["kia", "ev6", "niro"],
            "audi": ["audi", "e-tron"],
            "porsche": ["porsche", "taycan"],
            "bmw": ["bmw", "i3", "i4", "ix"]
        }

        # Check for EV entities in the text
        text_lower = text.lower()
        for brand, models in ev_entities.items():
            for entity in models:
                if entity in text_lower:
                    entities.append(entity)
                    # Also add the brand if we found a model
                    if entity != brand and brand not in entities:
                        entities.append(brand)

        # Add general EV terminology
        ev_terms = ["ev", "electric vehicle", "electric car", "battery electric",
                    "charging", "charger", "range", "battery"]

        for term in ev_terms:
            if term in text_lower:
                entities.append(term)

        return list(set(entities))  # Remove duplicates

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text

        Args:
            text (str): Input text

        Returns:
            dict: Sentiment analysis results
        """
        if not text or pd.isna(text) or len(text.strip()) < 10:
            return {"label": "neutral", "score": 0.5}

        # Make sure model is loaded
        if self.sentiment_pipeline is None:
            self.load_sentiment_model()

        try:
            # Limit to first 512 tokens due to model constraints
            result = self.sentiment_pipeline(text[:512])[0]

            # Map the label to positive/negative/neutral
            if result["label"].upper() == "POSITIVE" or result["label"].upper() == "LABEL_1":
                label = "positive"
                score = result["score"]
            elif result["label"].upper() == "NEGATIVE" or result["label"].upper() == "LABEL_0":
                label = "negative"
                score = result["score"]
            else:
                label = "neutral"
                score = 0.5

            # Apply threshold for more conservative labeling
            if label == "positive" and score < SENTIMENT_THRESHOLD:
                label = "neutral"
                score = 0.5
            elif label == "negative" and score < SENTIMENT_THRESHOLD:
                label = "neutral"
                score = 0.5

            return {"label": label, "score": score}

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"label": "neutral", "score": 0.5}

    def build_topic_model(self, texts, num_topics=None, preprocessing=True):
        """
        Build a topic model from texts

        Args:
            texts (list): List of text documents
            num_topics (int): Number of topics to extract
            preprocessing (bool): Whether to preprocess texts

        Returns:
            gensim.models.ldamodel.LdaModel: The trained topic model
        """
        logger.info("Building topic model...")
        n_topics = num_topics or NUM_TOPICS

        # Preprocess texts if needed
        processed_texts = []
        for text in tqdm(texts, desc="Preprocessing for topic modeling"):
            if not text or pd.isna(text):
                continue

            if preprocessing:
                processed_text = self.preprocess_text(text)
            else:
                processed_text = text

            processed_texts.append(processed_text)

        # Create tokenized texts
        tokenized_texts = [text.split() for text in processed_texts]

        # Create dictionary
        self.dictionary = corpora.Dictionary(tokenized_texts)

        # Filter extremes (optional)
        self.dictionary.filter_extremes(no_below=5, no_above=0.7, keep_n=100000)

        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_texts]

        # Build LDA model
        logger.info(f"Training LDA model with {n_topics} topics...")
        self.topic_model = gensim.models.ldamodel.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=n_topics,
            passes=10,
            alpha='auto',
            eta='auto',
            per_word_topics=True,
            random_state=42
        )

        # Evaluate model coherence
        try:
            coherence_model = CoherenceModel(
                model=self.topic_model,
                texts=tokenized_texts,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            coherence = coherence_model.get_coherence()
            logger.info(f"Topic model coherence score: {coherence:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute coherence score: {str(e)}")

        # Print top terms for each topic
        for idx, topic in self.topic_model.print_topics(num_words=5):
            logger.info(f"Topic {idx}: {topic}")

        return self.topic_model

    def assign_topics(self, text, threshold=0.2):
        """
        Assign topics to a given text

        Args:
            text (str): Input text
            threshold (float): Minimum probability threshold for topic assignment

        Returns:
            list: Assigned topics
        """
        if not text or pd.isna(text) or not self.topic_model:
            return []

        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Tokenize
        tokens = processed_text.split()

        # Get bow representation
        bow = self.dictionary.doc2bow(tokens)

        # Get topic distribution
        topic_dist = self.topic_model.get_document_topics(bow)

        # Filter topics above threshold
        relevant_topics = [topic_id for topic_id, prob in topic_dist if prob > threshold]

        # Map topic IDs to their top terms
        topic_terms = []
        for topic_id in relevant_topics:
            # Get the top terms for this topic
            terms = self.topic_model.show_topic(topic_id, 3)  # Get top 3 terms
            topic_term = "_".join([term for term, _ in terms])
            topic_terms.append(topic_term)

        return topic_terms

    def process_data(self, df, analyze_sentiment=True, extract_entities=True, assign_topics=False):
        """
        Process dataframe with all analysis steps

        Args:
            df (pandas.DataFrame): Input dataframe
            analyze_sentiment (bool): Whether to analyze sentiment
            extract_entities (bool): Whether to extract entities
            assign_topics (bool): Whether to assign topics

        Returns:
            pandas.DataFrame: Processed dataframe
        """
        logger.info(f"Processing {len(df)} records...")

        # Make sure sentiment model is loaded if needed
        if analyze_sentiment and self.sentiment_pipeline is None:
            self.load_sentiment_model()

        # Build topic model if needed
        if assign_topics and self.topic_model is None:
            self.build_topic_model(df['text'].tolist())

        # Create output dataframe
        result_df = df.copy()

        # Add columns for results
        if analyze_sentiment:
            result_df['sentiment'] = None
            result_df['sentiment_score'] = None

        if extract_entities:
            result_df['entities'] = None

        if assign_topics:
            result_df['topics'] = None

        # Process each record
        start_time = time.time()

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
            text = row['text']

            if analyze_sentiment:
                sentiment_result = self.analyze_sentiment(text)
                result_df.at[idx, 'sentiment'] = sentiment_result['label']
                result_df.at[idx, 'sentiment_score'] = sentiment_result['score']

            if extract_entities:
                entities = self.extract_entities(text)
                result_df.at[idx, 'entities'] = entities

            if assign_topics:
                topics = self.assign_topics(text)
                result_df.at[idx, 'topics'] = topics

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"Processed {len(df)} records in {processing_time:.2f} seconds")
        logger.info(f"Average processing time: {processing_time / len(df):.4f} seconds per record")

        return result_df

    def save_models(self, path_prefix=None):
        """
        Save all models to disk

        Args:
            path_prefix (str): Path prefix for saving models

        Returns:
            bool: True if successful
        """
        if path_prefix is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            path_prefix = os.path.join(MODELS_DIR, 'ev_classifier')

        logger.info(f"Saving models with prefix: {path_prefix}")

        try:
            # Save topic model components if available
            if self.topic_model:
                topic_model_path = f"{path_prefix}_topic_model.model"
                self.topic_model.save(topic_model_path)
                logger.info(f"Saved topic model to {topic_model_path}")

            if self.dictionary:
                dictionary_path = f"{path_prefix}_dictionary.pkl"
                with open(dictionary_path, 'wb') as f:
                    pickle.dump(self.dictionary, f)
                logger.info(f"Saved dictionary to {dictionary_path}")

            # Note: We don't save the sentiment model since it's a pre-trained model
            # that can be loaded from the transformers library

            logger.info("Models saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False

    def load_models(self, path_prefix=None):
        """
        Load all models from disk

        Args:
            path_prefix (str): Path prefix for loading models

        Returns:
            bool: True if successful
        """
        if path_prefix is None:
            path_prefix = os.path.join(MODELS_DIR, 'ev_classifier')

        logger.info(f"Loading models from prefix: {path_prefix}")

        try:
            # Load topic model components
            topic_model_path = f"{path_prefix}_topic_model.model"
            if os.path.exists(topic_model_path):
                self.topic_model = gensim.models.ldamodel.LdaModel.load(topic_model_path)
                logger.info(f"Loaded topic model from {topic_model_path}")

            dictionary_path = f"{path_prefix}_dictionary.pkl"
            if os.path.exists(dictionary_path):
                with open(dictionary_path, 'rb') as f:
                    self.dictionary = pickle.load(f)
                logger.info(f"Loaded dictionary from {dictionary_path}")

            # Load sentiment model
            self.load_sentiment_model()

            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False