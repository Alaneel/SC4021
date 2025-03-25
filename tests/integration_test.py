# tests/integration_test.py
"""Integration tests for the streaming opinions search engine."""

import unittest
import requests
import pandas as pd
import os
import sys
import time
import subprocess
import socket
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IntegrationTest(unittest.TestCase):
    """Integration tests for all components of the streaming opinions search engine."""

    @classmethod
    def setUpClass(cls):
        """Set up test class - check if Solr is running."""
        cls.solr_url = "http://localhost:8983/solr/streaming_opinions"
        cls.flask_url = "http://localhost:5000"
        cls.solr_running = cls.check_solr_running()

        # Check if processed data exists
        cls.data_path = Path("../data/processed_streaming_opinions.csv")
        cls.data_exists = cls.data_path.exists()

        # If Flask app is not running, start it in a subprocess
        cls.flask_process = None
        if not cls.check_flask_running():
            print("Starting Flask application...")
            try:
                # Start Flask app in a subprocess
                cls.flask_process = subprocess.Popen(
                    ["python", "../app.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                # Wait for Flask to start
                time.sleep(3)
                print("Flask application started")
            except Exception as e:
                print(f"Failed to start Flask app: {e}")

    @classmethod
    def tearDownClass(cls):
        """Tear down test class - stop Flask if we started it."""
        if cls.flask_process:
            print("Stopping Flask application...")
            cls.flask_process.terminate()
            cls.flask_process.wait()
            print("Flask application stopped")

    @classmethod
    def check_solr_running(cls):
        """Check if Solr is running."""
        try:
            response = requests.get(f"{cls.solr_url}/admin/ping", timeout=5)
            return response.status_code == 200
        except:
            return False

    @classmethod
    def check_flask_running(cls):
        """Check if Flask application is running."""
        try:
            response = requests.get(cls.flask_url, timeout=2)
            return response.status_code == 200
        except:
            return False

    def test_01_solr_connection(self):
        """Test connection to Solr."""
        if not self.solr_running:
            self.skipTest("Solr is not running")

        response = requests.get(f"{self.solr_url}/admin/ping", timeout=5)
        self.assertEqual(response.status_code, 200)

    def test_02_solr_schema(self):
        """Test Solr schema configuration."""
        if not self.solr_running:
            self.skipTest("Solr is not running")

        response = requests.get(f"{self.solr_url}/schema/fields", timeout=5)
        self.assertEqual(response.status_code, 200)

        fields = response.json().get('fields', [])
        field_names = [f['name'] for f in fields]

        # Check that core fields exist
        required_fields = ['id', 'text', 'platform', 'sentiment', 'sentiment_score']
        for field in required_fields:
            self.assertIn(field, field_names)

    def test_03_solr_documents(self):
        """Test that Solr contains documents."""
        if not self.solr_running:
            self.skipTest("Solr is not running")

        response = requests.get(f"{self.solr_url}/select?q=*:*&rows=0&wt=json", timeout=5)
        self.assertEqual(response.status_code, 200)

        num_found = response.json().get('response', {}).get('numFound', 0)
        self.assertGreater(num_found, 0)

    def test_04_flask_home_page(self):
        """Test Flask home page."""
        response = requests.get(f"{self.flask_url}/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Streaming Opinion Search", response.content)

    def test_05_flask_search_results(self):
        """Test search functionality."""
        if not self.solr_running:
            self.skipTest("Solr is not running")

        response = requests.get(f"{self.flask_url}/search?q=netflix")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Search Results", response.content)

    def test_06_flask_document_view(self):
        """Test document view functionality."""
        if not self.solr_running:
            self.skipTest("Solr is not running")

        # First get a document ID
        response = requests.get(f"{self.solr_url}/select?q=*:*&rows=1&wt=json", timeout=5)
        self.assertEqual(response.status_code, 200)

        docs = response.json().get('response', {}).get('docs', [])
        if not docs:
            self.skipTest("No documents found in Solr")

        doc_id = docs[0].get('id')

        # Now test document view
        response = requests.get(f"{self.flask_url}/document/{doc_id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Document Detail", response.content)

    def test_07_processed_data_exists(self):
        """Test that processed data exists."""
        if not self.data_exists:
            self.skipTest("Processed data file does not exist")

        self.assertTrue(self.data_path.exists())

    def test_08_data_processing_functions(self):
        """Test data processing functions."""
        if not self.data_exists:
            self.skipTest("Processed data file does not exist")

        # Import necessary modules
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from processing.processing import EnhancedDataProcessor

        # Create processor
        processor = EnhancedDataProcessor()

        # Test sentiment analysis function
        sentiment, score = processor.analyze_sentiment("I love netflix, it has great shows!")
        self.assertEqual(sentiment, "positive")
        self.assertGreater(score, 0)

        sentiment, score = processor.analyze_sentiment("I hate the new interface, it's terrible")
        self.assertEqual(sentiment, "negative")
        self.assertLess(score, 0)

        # Test platform detection
        platform = processor.detect_streaming_platform("Netflix has really improved their catalog")
        self.assertEqual(platform, "netflix")

        platform = processor.detect_streaming_platform("I prefer Disney+ for kids shows")
        self.assertEqual(platform, "disney+")

    def test_09_evaluation_setup(self):
        """Test evaluation framework setup."""
        try:
            from evaluation.evaluate_classifier import ClassifierEvaluator

            # Create evaluator
            evaluator = ClassifierEvaluator()

            # Test precision calculation with dummy data
            from sklearn.metrics import precision_recall_fscore_support

            y_true = ['positive', 'negative', 'positive', 'neutral', 'negative']
            y_pred = ['positive', 'negative', 'positive', 'negative', 'negative']

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )

            self.assertGreater(precision, 0)
            self.assertGreater(recall, 0)
            self.assertGreater(f1, 0)

        except ImportError:
            self.skipTest("Evaluation framework not found")

    def test_10_performance_metrics(self):
        """Test performance metrics framework."""
        try:
            from evaluation.performance_metrics import PerformanceEvaluator

            # Create evaluator
            evaluator = PerformanceEvaluator()

            # Add a test query
            evaluator.add_search_query("*:*", "All documents")

            # Only run if Solr is available
            if self.solr_running:
                # Run the query once
                results = evaluator.run_queries(repeats=1)

                # Check that we have a result
                self.assertIn(1, results)
                self.assertIn('query', results[1])
                self.assertIn('avg_time_ms', results[1])

        except ImportError:
            self.skipTest("Performance metrics framework not found")


if __name__ == "__main__":
    unittest.main()