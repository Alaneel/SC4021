# advanced_features/interactive_feedback.py
"""Interactive search feedback module for refining search results."""

import requests
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class InteractiveFeedback:
    """Interactive feedback system that refines search results based on user judgments."""

    def __init__(self, solr_url="http://localhost:8983/solr/streaming_opinions"):
        """
        Initialize the interactive feedback system.

        Args:
            solr_url (str): URL of the Solr instance
        """
        self.solr_url = solr_url
        self.relevant_docs = []
        self.irrelevant_docs = []
        self.original_query = None
        self.expanded_terms = []
        self.vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english')
        self.doc_vectors = None
        self.all_docs = []

    def set_query(self, query):
        """
        Set the original query.

        Args:
            query (str): The original search query

        Returns:
            self: For method chaining
        """
        self.original_query = query
        self.relevant_docs = []
        self.irrelevant_docs = []
        self.expanded_terms = []
        return self

    def get_initial_results(self, rows=20, filters=None):
        """
        Get initial search results.

        Args:
            rows (int): Number of results to return
            filters (list): List of filter queries

        Returns:
            list: Search results
        """
        if not self.original_query:
            raise ValueError("Query must be set before getting results")

        # Default params
        params = {
            'q': self.original_query,
            'rows': rows,
            'fl': 'id,text,cleaned_text,title,platform,sentiment,score',
            'wt': 'json'
        }

        # Add filters if provided
        if filters:
            params['fq'] = filters

        try:
            response = requests.get(f"{self.solr_url}/select", params=params)
            response.raise_for_status()
            results = response.json()

            # Store all docs for later use
            self.all_docs = results['response']['docs']

            return self.all_docs
        except Exception as e:
            print(f"Error getting search results: {e}")
            return []

    def add_relevant_document(self, doc_id):
        """
        Add a document to the set of relevant documents.

        Args:
            doc_id (str): ID of the relevant document

        Returns:
            self: For method chaining
        """
        # Find the document in all_docs
        doc = next((d for d in self.all_docs if d['id'] == doc_id), None)

        if doc and doc_id not in self.relevant_docs:
            self.relevant_docs.append(doc_id)

            # Remove from irrelevant if it was there
            if doc_id in self.irrelevant_docs:
                self.irrelevant_docs.remove(doc_id)

        return self

    def add_irrelevant_document(self, doc_id):
        """
        Add a document to the set of irrelevant documents.

        Args:
            doc_id (str): ID of the irrelevant document

        Returns:
            self: For method chaining
        """
        if doc_id not in self.irrelevant_docs:
            self.irrelevant_docs.append(doc_id)

            # Remove from relevant if it was there
            if doc_id in self.relevant_docs:
                self.relevant_docs.remove(doc_id)

        return self

    def extract_key_terms(self, doc_ids, n_terms=5):
        """
        Extract key terms from documents.

        Args:
            doc_ids (list): List of document IDs
            n_terms (int): Number of key terms to extract

        Returns:
            list: List of key terms with scores
        """
        if not doc_ids:
            return []

        # Get the documents
        docs = [d for d in self.all_docs if d['id'] in doc_ids]

        # Extract text content
        texts = []
        for doc in docs:
            # Prefer cleaned text if available
            if 'cleaned_text' in doc and doc['cleaned_text']:
                texts.append(doc['cleaned_text'])
            elif 'text' in doc and doc['text']:
                texts.append(doc['text'])

        if not texts:
            return []

        # Fit vectorizer and transform texts
        try:
            X = self.vectorizer.fit_transform(texts)

            # Get feature names and their importance
            feature_names = np.array(self.vectorizer.get_feature_names_out())

            # Sum TF-IDF scores across documents
            tfidf_sum = X.sum(axis=0).A1

            # Get top terms
            top_indices = tfidf_sum.argsort()[-n_terms:][::-1]
            top_terms = [(feature_names[i], tfidf_sum[i]) for i in top_indices]

            return top_terms
        except Exception as e:
            print(f"Error extracting key terms: {e}")
            return []

    def expand_query(self, use_relevance_feedback=True, max_terms=3):
        """
        Expand the original query using relevance feedback.

        Args:
            use_relevance_feedback (bool): Whether to use relevance feedback
            max_terms (int): Maximum number of terms to add

        Returns:
            str: Expanded query
        """
        if not self.original_query:
            raise ValueError("Query must be set before expanding")

        if not use_relevance_feedback or not self.relevant_docs:
            return self.original_query

        # Extract key terms from relevant documents
        key_terms = self.extract_key_terms(self.relevant_docs)

        if not key_terms:
            return self.original_query

        # Extract key terms from irrelevant documents to avoid
        negative_terms = []
        if self.irrelevant_docs:
            negative_terms = self.extract_key_terms(self.irrelevant_docs)
            negative_terms = [term[0] for term in negative_terms]

        # Filter out terms that appear in irrelevant docs
        filtered_terms = [term for term in key_terms if term[0] not in negative_terms]

        # Take top terms
        top_terms = filtered_terms[:max_terms]

        # Create expanded query
        expansion_text = " ".join([f"{term[0]}^{round(term[1], 2)}" for term in top_terms])
        expanded_query = f"{self.original_query} {expansion_text}"

        # Store expanded terms
        self.expanded_terms = [term[0] for term in top_terms]

        return expanded_query

    def get_refined_results(self, rows=20, filters=None, use_relevance_feedback=True):
        """
        Get refined search results using relevance feedback.

        Args:
            rows (int): Number of results to return
            filters (list): List of filter queries
            use_relevance_feedback (bool): Whether to use relevance feedback

        Returns:
            tuple: (results, expanded_query)
        """
        # Expand query
        expanded_query = self.expand_query(use_relevance_feedback)

        # Default params
        params = {
            'q': expanded_query,
            'rows': rows,
            'fl': 'id,text,cleaned_text,title,platform,sentiment,score',
            'wt': 'json'
        }

        # Add filters if provided
        if filters:
            params['fq'] = filters

        try:
            response = requests.get(f"{self.solr_url}/select", params=params)
            response.raise_for_status()
            results = response.json()

            return results['response']['docs'], expanded_query
        except Exception as e:
            print(f"Error getting refined results: {e}")
            return [], expanded_query

    def find_similar_documents(self, doc_id, num_similar=5):
        """
        Find documents similar to a given document.

        Args:
            doc_id (str): Document ID to find similar documents for
            num_similar (int): Number of similar documents to return

        Returns:
            list: Similar documents with similarity scores
        """
        # Get the document
        doc = next((d for d in self.all_docs if d['id'] == doc_id), None)

        if not doc:
            return []

        # Get text content
        doc_text = ""
        if 'cleaned_text' in doc and doc['cleaned_text']:
            doc_text = doc['cleaned_text']
        elif 'text' in doc and doc['text']:
            doc_text = doc['text']

        if not doc_text:
            return []

        # Get text from all documents
        all_texts = []
        for d in self.all_docs:
            if 'cleaned_text' in d and d['cleaned_text']:
                all_texts.append(d['cleaned_text'])
            elif 'text' in d and d['text']:
                all_texts.append(d['text'])
            else:
                all_texts.append("")

        # Vectorize texts
        try:
            X = self.vectorizer.fit_transform(all_texts)

            # Get index of the document
            doc_index = next((i for i, d in enumerate(self.all_docs) if d['id'] == doc_id), -1)

            if doc_index == -1:
                return []

            # Calculate similarity
            doc_vector = X[doc_index:doc_index + 1]
            similarities = cosine_similarity(doc_vector, X).flatten()

            # Get top similar documents (excluding self)
            similar_indices = similarities.argsort()[:-num_similar - 2:-1]

            # Skip the document itself
            similar_indices = [i for i in similar_indices if i != doc_index]

            # Get similar documents with scores
            similar_docs = []
            for i in similar_indices[:num_similar]:
                similar_docs.append({
                    'doc': self.all_docs[i],
                    'similarity': float(similarities[i])
                })

            return similar_docs
        except Exception as e:
            print(f"Error finding similar documents: {e}")
            return []

    def get_filter_suggestions(self):
        """
        Generate filter suggestions based on relevant documents.

        Returns:
            dict: Suggested filters
        """
        if not self.relevant_docs:
            return {}

        # Get the relevant documents
        docs = [d for d in self.all_docs if d['id'] in self.relevant_docs]

        if not docs:
            return {}

        # Count platforms
        platforms = {}
        for doc in docs:
            if 'platform' in doc:
                platform = doc['platform']
                platforms[platform] = platforms.get(platform, 0) + 1

        # Count sentiments
        sentiments = {}
        for doc in docs:
            if 'sentiment' in doc:
                sentiment = doc['sentiment']
                sentiments[sentiment] = sentiments.get(sentiment, 0) + 1

        # Calculate most common filters
        suggestions = {}

        if platforms:
            # Get platform with highest count
            top_platform = max(platforms.items(), key=lambda x: x[1])
            if top_platform[1] / len(docs) >= 0.5:  # If at least half the docs have this platform
                suggestions['platform'] = top_platform[0]

        if sentiments:
            # Get sentiment with highest count
            top_sentiment = max(sentiments.items(), key=lambda x: x[1])
            if top_sentiment[1] / len(docs) >= 0.5:  # If at least half the docs have this sentiment
                suggestions['sentiment'] = top_sentiment[0]

        return suggestions

    def get_feedback_stats(self):
        """
        Get statistics about the feedback session.

        Returns:
            dict: Feedback statistics
        """
        stats = {
            'original_query': self.original_query,
            'expanded_terms': self.expanded_terms,
            'num_relevant': len(self.relevant_docs),
            'num_irrelevant': len(self.irrelevant_docs),
            'filter_suggestions': self.get_filter_suggestions()
        }

        return stats


# Example function for integrating with Flask app
def apply_interactive_feedback(app, solr_url):
    """
    Apply interactive feedback to a Flask app.

    Args:
        app: Flask application
        solr_url (str): URL of the Solr instance
    """
    from flask import request, jsonify, session

    feedback_system = InteractiveFeedback(solr_url)

    @app.route('/api/feedback/init', methods=['POST'])
    def init_feedback():
        """Initialize a feedback session."""
        data = request.json
        query = data.get('query')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Store in session
        session['feedback_query'] = query

        # Initialize feedback system
        feedback_system.set_query(query)
        results = feedback_system.get_initial_results()

        return jsonify({
            'status': 'success',
            'results': results,
            'stats': {
                'num_results': len(results),
                'original_query': query
            }
        })

    @app.route('/api/feedback/add', methods=['POST'])
    def add_feedback():
        """Add relevance feedback for a document."""
        data = request.json
        doc_id = data.get('doc_id')
        relevant = data.get('relevant', True)

        if not doc_id:
            return jsonify({'error': 'Document ID is required'}), 400

        # Make sure we have a query
        query = session.get('feedback_query')
        if not query:
            return jsonify({'error': 'No active feedback session'}), 400

        # Add feedback
        if relevant:
            feedback_system.add_relevant_document(doc_id)
        else:
            feedback_system.add_irrelevant_document(doc_id)

        # Get stats
        stats = feedback_system.get_feedback_stats()

        return jsonify({
            'status': 'success',
            'stats': stats
        })

    @app.route('/api/feedback/refine', methods=['GET'])
    def refine_results():
        """Get refined search results based on feedback."""
        # Make sure we have a query
        query = session.get('feedback_query')
        if not query:
            return jsonify({'error': 'No active feedback session'}), 400

        # Get refined results
        results, expanded_query = feedback_system.get_refined_results()

        return jsonify({
            'status': 'success',
            'results': results,
            'expanded_query': expanded_query,
            'stats': feedback_system.get_feedback_stats()
        })

    @app.route('/api/feedback/similar/<doc_id>', methods=['GET'])
    def get_similar_docs(doc_id):
        """Get documents similar to a given document."""
        # Get similar documents
        similar_docs = feedback_system.find_similar_documents(doc_id)

        return jsonify({
            'status': 'success',
            'similar_docs': similar_docs
        })

    @app.route('/api/feedback/reset', methods=['POST'])
    def reset_feedback():
        """Reset the feedback session."""
        # Clear session
        session.pop('feedback_query', None)

        return jsonify({
            'status': 'success'
        })