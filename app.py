from flask import Flask, render_template, request, jsonify
import requests
import json
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re

app = Flask(__name__)

# Solr connection settings
SOLR_URL = "http://localhost:8983/solr/streaming_opinions"


@app.route('/')
def index():
    """Render the search page"""
    # Get list of platforms for filtering
    platforms = get_platforms()
    # Get list of content types
    content_types = get_content_types()
    return render_template('index.html', platforms=platforms, content_types=content_types)


@app.route('/search', methods=['GET', 'POST'])
def search():
    """Handle search requests"""
    # Get search parameters
    query = request.args.get('q', '*:*')
    platform = request.args.get('platform', '')
    content_type = request.args.get('type', '')
    sentiment = request.args.get('sentiment', '')
    feature = request.args.get('feature', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    rows = int(request.args.get('rows', 10))
    start = int(request.args.get('start', 0))

    # Build filter queries
    fq = []
    if platform:
        fq.append(f'platform:"{platform}"')
    
    if content_type:
        fq.append(f'type:"{content_type}"')

    if sentiment:
        fq.append(f'sentiment:"{sentiment}"')
        
    # Add feature filter if selected
    if feature and feature != 'any':
        fq.append(f'{feature}:[0.5 TO *]')

    if start_date and end_date:
        fq.append(f'created_at:[{start_date}T00:00:00Z TO {end_date}T23:59:59Z]')

    # Build the Solr query parameters
    params = {
        'q': query,
        'fq': fq,
        'rows': rows,
        'start': start,
        'sort': 'score desc',
        'fl': 'id,title,text,full_text,cleaned_text,cleaned_full_text,platform,source,subreddit,created_at,sentiment,sentiment_score,score,author,permalink,content_quality,pricing,ui_ux,technical,customer_service,type,parent_id,num_comments,word_count',
        'wt': 'json'
    }

    # Send request to Solr
    response = requests.get(f"{SOLR_URL}/select", params=params)
    results = response.json()

    # Get facet information for visualizations
    facet_params = {
        'q': query,
        'fq': fq,
        'facet': 'true',
        'facet.field': ['platform', 'sentiment', 'source', 'type'],
        'facet.range': 'created_at',
        'facet.range.start': 'NOW-1YEAR',
        'facet.range.end': 'NOW',
        'facet.range.gap': '+1MONTH',
        'rows': 0,
        'wt': 'json'
    }

    facet_response = requests.get(f"{SOLR_URL}/select", params=facet_params)
    facets = facet_response.json().get('facet_counts', {})

    # Generate visualizations
    visualizations = generate_visualizations(facets, query)

    # Get filtering options
    platforms = get_platforms()
    content_types = get_content_types()
    features = [
        {'id': 'content_quality', 'name': 'Content Quality'},
        {'id': 'pricing', 'name': 'Pricing'},
        {'id': 'ui_ux', 'name': 'UI/UX'},
        {'id': 'technical', 'name': 'Technical'},
        {'id': 'customer_service', 'name': 'Customer Service'},
        {'id': 'any', 'name': 'Any Feature'}
    ]

    return render_template(
        'search_results.html',
        query=query,
        results=results['response']['docs'],
        num_found=results['response']['numFound'],
        start=start,
        rows=rows,
        platforms=platforms,
        content_types=content_types,
        features=features,
        selected_platform=platform,
        selected_type=content_type,
        selected_sentiment=sentiment,
        selected_feature=feature,
        start_date=start_date,
        end_date=end_date,
        visualizations=visualizations
    )


@app.route('/api/sentiment')
def sentiment_api():
    """API endpoint for sentiment analysis"""
    platform = request.args.get('platform', '')
    content_type = request.args.get('type', '')
    timeframe = request.args.get('timeframe', '1y')  # Options: 1m, 3m, 6m, 1y, all

    # Calculate date range based on timeframe
    end_date = datetime.now()

    if timeframe == '1m':
        start_date = end_date - timedelta(days=30)
    elif timeframe == '3m':
        start_date = end_date - timedelta(days=90)
    elif timeframe == '6m':
        start_date = end_date - timedelta(days=180)
    elif timeframe == '1y':
        start_date = end_date - timedelta(days=365)
    else:  # all
        start_date = datetime(2010, 1, 1)  # Far enough in the past

    # Convert to Solr date format
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Build filter queries
    fq = [f'created_at:[{start_date_str} TO {end_date_str}]']
    if platform:
        fq.append(f'platform:"{platform}"')
    if content_type:
        fq.append(f'type:"{content_type}"')

    # Get sentiment counts
    params = {
        'q': '*:*',
        'fq': fq,
        'facet': 'true',
        'facet.field': 'sentiment',
        'rows': 0,
        'wt': 'json'
    }

    response = requests.get(f"{SOLR_URL}/select", params=params)
    facets = response.json().get('facet_counts', {}).get('facet_fields', {})

    # Process sentiment data
    sentiment_data = {}
    if 'sentiment' in facets:
        sentiment_facets = facets['sentiment']
        for i in range(0, len(sentiment_facets), 2):
            if i + 1 < len(sentiment_facets):
                sentiment_data[sentiment_facets[i]] = sentiment_facets[i + 1]

    return jsonify(sentiment_data)


@app.route('/api/features')
def features_api():
    """API endpoint for feature scores"""
    platform = request.args.get('platform', '')
    timeframe = request.args.get('timeframe', '1y')  # Options: 1m, 3m, 6m, 1y, all

    # Calculate date range based on timeframe
    end_date = datetime.now()

    if timeframe == '1m':
        start_date = end_date - timedelta(days=30)
    elif timeframe == '3m':
        start_date = end_date - timedelta(days=90)
    elif timeframe == '6m':
        start_date = end_date - timedelta(days=180)
    elif timeframe == '1y':
        start_date = end_date - timedelta(days=365)
    else:  # all
        start_date = datetime(2010, 1, 1)  # Far enough in the past

    # Convert to Solr date format
    start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Build filter queries
    fq = [f'created_at:[{start_date_str} TO {end_date_str}]']
    if platform:
        fq.append(f'platform:"{platform}"')

    # Get feature stats
    params = {
        'q': '*:*',
        'fq': fq,
        'stats': 'true',
        'stats.field': ['content_quality', 'pricing', 'ui_ux', 'technical', 'customer_service'],
        'rows': 0,
        'wt': 'json'
    }

    response = requests.get(f"{SOLR_URL}/select", params=params)
    stats = response.json().get('stats', {}).get('stats_fields', {})

    # Process feature data
    feature_data = {}
    for feature in stats:
        feature_data[feature] = {
            'mean': stats[feature].get('mean', 0),
            'count': stats[feature].get('count', 0)
        }

    return jsonify(feature_data)


@app.route('/document/<doc_id>')
def view_document(doc_id):
    """View a single document"""
    # Get document from Solr
    params = {
        'q': f'id:"{doc_id}"',
        'wt': 'json'
    }

    response = requests.get(f"{SOLR_URL}/select", params=params)
    results = response.json()

    if results['response']['numFound'] > 0:
        doc = results['response']['docs'][0]
        
        # Get related documents (same thread) if it's a comment
        related_docs = []
        if 'parent_id' in doc and doc.get('parent_id'):
            parent_params = {
                'q': f'id:"{doc["parent_id"].replace("t1_", "").replace("t3_", "")}"',
                'wt': 'json'
            }
            parent_response = requests.get(f"{SOLR_URL}/select", params=parent_params)
            parent_results = parent_response.json()
            
            if parent_results['response']['numFound'] > 0:
                related_docs.append(parent_results['response']['docs'][0])
                
        # Get keywords as a list
        keywords = []
        if 'keywords' in doc:
            if isinstance(doc['keywords'], list):
                keywords = doc['keywords']
            elif isinstance(doc['keywords'], str) and doc['keywords'].startswith('['):
                try:
                    keywords = json.loads(doc['keywords'].replace("'", '"'))
                except:
                    pass
                    
        # Get entities as a list
        entities = []
        if 'entities' in doc:
            if isinstance(doc['entities'], list):
                entities = doc['entities']
            elif isinstance(doc['entities'], str) and doc['entities'].startswith('['):
                try:
                    entities = json.loads(doc['entities'].replace("'", '"'))
                except:
                    pass
                    
        return render_template('document.html', doc=doc, related_docs=related_docs, 
                              keywords=keywords, entities=entities)
    else:
        return render_template('error.html', message="Document not found"), 404


def get_platforms():
    """Get list of platforms from Solr"""
    params = {
        'q': '*:*',
        'facet': 'true',
        'facet.field': 'platform',
        'facet.limit': 20,
        'rows': 0,
        'wt': 'json'
    }

    response = requests.get(f"{SOLR_URL}/select", params=params)
    facets = response.json().get('facet_counts', {}).get('facet_fields', {})

    platforms = []
    if 'platform' in facets:
        platform_facets = facets['platform']
        for i in range(0, len(platform_facets), 2):
            if i + 1 < len(platform_facets) and platform_facets[i + 1] > 0:
                platforms.append(platform_facets[i])

    return platforms


def get_content_types():
    """Get list of content types from Solr"""
    params = {
        'q': '*:*',
        'facet': 'true',
        'facet.field': 'type',
        'facet.limit': 10,
        'rows': 0,
        'wt': 'json'
    }

    response = requests.get(f"{SOLR_URL}/select", params=params)
    facets = response.json().get('facet_counts', {}).get('facet_fields', {})

    types = []
    if 'type' in facets:
        type_facets = facets['type']
        for i in range(0, len(type_facets), 2):
            if i + 1 < len(type_facets) and type_facets[i + 1] > 0:
                types.append(type_facets[i])

    return types


def generate_visualizations(facets, query):
    """Generate visualizations from facet data"""
    visualizations = {}

    # Platform distribution pie chart
    if 'facet_fields' in facets and 'platform' in facets['facet_fields']:
        platform_data = []
        platform_facets = facets['facet_fields']['platform']

        for i in range(0, len(platform_facets), 2):
            if i + 1 < len(platform_facets) and platform_facets[i + 1] > 0:
                platform_data.append({
                    'platform': platform_facets[i],
                    'count': platform_facets[i + 1]
                })

        if platform_data:
            df = pd.DataFrame(platform_data)
            fig = px.pie(df, values='count', names='platform',
                         title=f'Platform Distribution for query: {query}')
            visualizations['platform_pie'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Content type distribution pie chart
    if 'facet_fields' in facets and 'type' in facets['facet_fields']:
        type_data = []
        type_facets = facets['facet_fields']['type']

        for i in range(0, len(type_facets), 2):
            if i + 1 < len(type_facets) and type_facets[i + 1] > 0:
                type_data.append({
                    'type': type_facets[i],
                    'count': type_facets[i + 1]
                })

        if type_data:
            df = pd.DataFrame(type_data)
            fig = px.pie(df, values='count', names='type',
                         title=f'Content Type Distribution for query: {query}')
            visualizations['type_pie'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Sentiment distribution bar chart
    if 'facet_fields' in facets and 'sentiment' in facets['facet_fields']:
        sentiment_data = []
        sentiment_facets = facets['facet_fields']['sentiment']

        for i in range(0, len(sentiment_facets), 2):
            if i + 1 < len(sentiment_facets) and sentiment_facets[i + 1] > 0:
                sentiment_data.append({
                    'sentiment': sentiment_facets[i],
                    'count': sentiment_facets[i + 1]
                })

        if sentiment_data:
            df = pd.DataFrame(sentiment_data)
            fig = px.bar(df, x='sentiment', y='count', color='sentiment',
                         title=f'Sentiment Distribution for query: {query}')
            visualizations['sentiment_bar'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Time series chart
    if 'facet_ranges' in facets and 'created_at' in facets['facet_ranges']:
        time_data = []
        counts = facets['facet_ranges']['created_at']['counts']

        for i in range(0, len(counts), 2):
            if i + 1 < len(counts):
                date_str = counts[i]
                count = counts[i + 1]

                # Parse date from Solr format
                date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
                month_year = date.strftime('%b %Y')

                time_data.append({
                    'month': month_year,
                    'count': count,
                    'date': date  # For sorting
                })

        if time_data:
            df = pd.DataFrame(time_data)
            df = df.sort_values('date')

            fig = px.line(df, x='month', y='count', markers=True,
                          title=f'Post Volume Over Time for query: {query}')
            visualizations['time_series'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return visualizations


if __name__ == '__main__':
    app.run(debug=True, port=5000)