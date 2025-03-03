"""
Flask web application for EV Opinion Search Engine
"""
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import os
import json
import logging
import sys
from datetime import datetime

from config.app_config import (
    FLASK_SECRET_KEY,
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    PROCESSED_DATA_DIR
)
from indexing.search_utils import execute_search, format_search_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY


@app.route('/')
def index():
    """Render the main search page"""
    return render_template('index.html')


@app.route('/search')
def search():
    """Handle search requests"""
    # Get search parameters
    query = request.args.get('q', '*:*')
    start = int(request.args.get('start', 0))
    rows = int(request.args.get('rows', 10))
    sort = request.args.get('sort', 'score desc')

    # Optional filters
    filters = []

    # Date range filter
    from_date = request.args.get('from')
    to_date = request.args.get('to')
    if from_date and to_date:
        date_filter = f"created_utc:[{from_date}T00:00:00Z TO {to_date}T23:59:59Z]"
        filters.append(date_filter)

    # Sentiment filter
    sentiment = request.args.get('sentiment')
    if sentiment:
        filters.append(f"sentiment:{sentiment}")

    # Subreddit filter
    subreddit = request.args.get('subreddit')
    if subreddit:
        filters.append(f"subreddit:{subreddit}")

    # Topic filter
    topic = request.args.get('topic')
    if topic:
        filters.append(f"topics:{topic}")

    # Entity filter
    entity = request.args.get('entity')
    if entity:
        filters.append(f"entities:{entity}")

    # Content type filter
    content_type = request.args.get('type')
    if content_type:
        filters.append(f"type:{content_type}")

    # Facet fields
    facets = [
        'sentiment',
        'subreddit',
        'topics',
        'entities',
        'type'
    ]

    # Execute search
    search_results = execute_search(
        query=query,
        filters=filters,
        facets=facets,
        start=start,
        rows=rows,
        sort=sort
    )

    # Format results and add visualizations
    formatted_results = format_search_results(search_results)

    # Return JSON response
    return jsonify(formatted_results)


@app.route('/suggest')
def suggest():
    """Return search suggestions"""
    query = request.args.get('q', '')

    if not query or len(query) < 2:
        return jsonify([])

    # Execute simple search with ngram field
    search_results = execute_search(
        query=f"{query}*",
        filters=None,
        facets=None,
        start=0,
        rows=5,
        sort=None
    )

    # Extract suggestions from results
    suggestions = []

    for doc in search_results.get('docs', []):
        # Add title if available
        if 'title' in doc and doc['title'] and len(doc['title']) > 0:
            if query.lower() in doc['title'].lower():
                suggestions.append(doc['title'])

        # Extract meaningful phrases from text
        if 'text' in doc and doc['text']:
            text = doc['text']

            # Find position of query in text
            pos = text.lower().find(query.lower())
            if pos >= 0:
                # Extract a snippet around the query
                start = max(0, pos - 30)
                end = min(len(text), pos + 30)

                # Find word boundaries
                while start > 0 and text[start] != ' ':
                    start -= 1

                while end < len(text) and text[end] != ' ':
                    end += 1

                snippet = text[start:end].strip()
                if snippet and len(snippet) > len(query) + 10:
                    suggestions.append(snippet)

    # Remove duplicates and limit to 5
    suggestions = list(set(suggestions))[:5]

    return jsonify(suggestions)


@app.route('/facets')
def facets():
    """Return facet information for visualization"""
    # Execute search to get facets
    search_results = execute_search(
        query='*:*',
        filters=None,
        facets=['sentiment', 'subreddit', 'topics', 'entities', 'type'],
        start=0,
        rows=0
    )

    # Extract facets
    if 'facets' in search_results:
        return jsonify(search_results['facets'])
    else:
        return jsonify({})


@app.route('/stats')
def stats():
    """Return statistics about the indexed data"""
    # Execute search to get stats
    search_results = execute_search(
        query='*:*',
        filters=None,
        facets=['sentiment', 'type'],
        start=0,
        rows=0
    )

    # Calculate basic stats
    total_docs = search_results.get('num_found', 0)

    stats = {
        'total_documents': total_docs,
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Add sentiment distribution if available
    if ('facets' in search_results and
            'facet_fields' in search_results['facets'] and
            'sentiment' in search_results['facets']['facet_fields']):
        sentiment_data = dict(zip(
            search_results['facets']['facet_fields']['sentiment'][::2],
            search_results['facets']['facet_fields']['sentiment'][1::2]
        ))

        stats['sentiment_distribution'] = sentiment_data

    # Add content type distribution if available
    if ('facets' in search_results and
            'facet_fields' in search_results['facets'] and
            'type' in search_results['facets']['facet_fields']):
        type_data = dict(zip(
            search_results['facets']['facet_fields']['type'][::2],
            search_results['facets']['facet_fields']['type'][1::2]
        ))

        stats['content_types'] = type_data

    return jsonify(stats)


@app.route('/about')
def about():
    """Show information about the project"""
    return render_template('about.html')


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)