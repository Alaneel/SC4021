from flask import Flask, render_template, request, jsonify
import requests
import json
import pandas as pd
import plotly
import plotly.express as px
import logging
import sys
from datetime import datetime, timedelta

app = Flask(__name__)

# Solr connection settings
SOLR_URL = "http://localhost:8983/solr/streaming_opinions"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Test Solr on startup
def test_solr_connection():
    try:
        response = requests.get(f"{SOLR_URL}/admin/ping", timeout=5)
        response.raise_for_status()
        logger.info(f"Successfully connected to Solr at {SOLR_URL}")

        # Check if there are any documents
        count_response = requests.get(f"{SOLR_URL}/select?q=*:*&rows=0&wt=json", timeout=5)
        count_data = count_response.json()
        doc_count = count_data.get("response", {}).get("numFound", 0)
        logger.info(f"Found {doc_count} documents in Solr index")

        return True
    except Exception as e:
        logger.error(f"Failed to connect to Solr: {e}")
        return False

@app.route('/')
def index():
    """Render the search page"""
    # Get actual platforms from Solr
    platforms = get_facet_values('platform')
    content_types = get_facet_values('type')

    return render_template('index.html', platforms=platforms, content_types=content_types)


def get_facet_values(field_name):
    """Get facet values from Solr"""
    params = {
        'q': '*:*',
        'facet': 'true',
        'facet.field': field_name,
        'facet.mincount': 1,
        'facet.limit': 20,
        'rows': 0,
        'wt': 'json'
    }

    try:
        response = requests.get(f"{SOLR_URL}/select", params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        facets = response.json().get('facet_counts', {}).get('facet_fields', {})

        values = []
        if field_name in facets:
            field_facets = facets[field_name]
            for i in range(0, len(field_facets), 2):
                if i + 1 < len(field_facets) and field_facets[i + 1] > 0:
                    values.append(field_facets[i])

        return values
    except Exception as e:
        print(f"Error fetching {field_name} facets: {e}")
        return []  # Return empty list on error


@app.route('/search')
def search():
    """Handle search requests with consistent query handling"""
    # Get the raw query directly from request args - store original form
    raw_query = request.args.get('q', '*:*')
    original_query = raw_query  # Store the original query for pagination links

    platform = request.args.get('platform', '')
    content_type = request.args.get('type', '')
    sentiment = request.args.get('sentiment', '')
    feature = request.args.get('feature', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    rows = int(request.args.get('rows', 10))
    start = int(request.args.get('start', 0))

    # Get sort parameter, default to 'score desc'
    sort = request.args.get('sort', 'score desc')

    # Build filter queries with any necessary escaping
    fq = []
    if platform:
        fq.append(f'platform:"{platform}"')

    if content_type:
        fq.append(f'type:"{content_type}"')

    if sentiment:
        fq.append(f'sentiment:"{sentiment}"')

    if feature and feature != 'any':
        fq.append(f'{feature}:[0.5 TO *]')

    '''
    if start_date and end_date:
        fq.append(f'created_at:[{start_date}T00:00:00Z TO {end_date}T23:59:59Z]')
    '''

    # Modified date filtering to handle start_date and/or end_date
    if start_date or end_date:
        # Default to unbounded ranges if one date is missing
        start_range = start_date + 'T00:00:00Z' if start_date else '*'
        end_range = end_date + 'T23:59:59Z' if end_date else '*'
        fq.append(f'created_at:[{start_range} TO {end_range}]')

    # Process the query - don't expand it unless it's the special wildcard query
    solr_query = raw_query
    if raw_query != '*:*' and not raw_query.startswith('_text_:'):
        # Use a simpler transformation that's less prone to expanding with each page
        solr_query = f'text:"{raw_query}" OR title:"{raw_query}"'

    # Request all necessary schema fields
    params = {
        'q': solr_query,
        'fq': fq,
        'rows': rows,
        'start': start,
        'facet': 'true',
        'facet.field': ['platform', 'sentiment', 'type'],
        'facet.mincount': 1,
        'facet.range': 'created_at',
        'facet.range.start': 'NOW-1YEAR',
        'facet.range.end': 'NOW',
        'facet.range.gap': '+1MONTH',
        'sort': sort,
        'fl': 'id,text,cleaned_text,full_text,cleaned_full_text,title,platform,source,created_at,score,type,author,subreddit,permalink,parent_id,num_comments,word_count,sentiment,sentiment_score,subjectivity_score,content_quality,pricing,ui_ux,technical,customer_service',
        'wt': 'json',
    }

    try:
        print(f"Solr Query: {solr_query}")
        print(f"Filter Queries: {fq}")

        response = requests.get(f"{SOLR_URL}/select", params=params)
        response.raise_for_status()
        results = response.json()

        # Get actual number of results
        num_found = results['response'].get('numFound', 0)

        # Handle pagination edge cases
        if num_found == 0:
            # No results found, but we'll keep the start parameter
            # to maintain navigation capability
            pass
        elif start >= num_found:
            # Start is beyond available results, go to last page
            start = (num_found // rows) * rows
            if start == num_found:  # Handle exact division case
                start = max(0, start - rows)

            # Re-fetch with corrected pagination
            params['start'] = start
            response = requests.get(f"{SOLR_URL}/select", params=params)
            response.raise_for_status()
            results = response.json()

        # Generate visualizations from facet data
        visualizations = {}
        if 'facet_counts' in results:
            visualizations = generate_visualizations(results['facet_counts'], raw_query)

        # Get filter options for sidebar
        platforms = get_facet_values('platform')
        content_types = get_facet_values('type')

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
            query=original_query,  # Use the original query for display and pagination links
            solr_query=solr_query,  # The actual query sent to Solr
            results=results['response']['docs'],
            num_found=num_found,
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
            visualizations=visualizations,
            sort=sort
        )
    except Exception as e:
        error_message = f"Error searching Solr: {str(e)}"
        print(error_message)
        return render_template('error.html', message=error_message), 500


@app.route('/document/<doc_id>')
def view_document(doc_id):
    """View a single document with proper schema fields"""
    params = {
        'q': f'id:"{doc_id}"',
        'fl': '*',  # Request all fields
        'wt': 'json'
    }

    try:
        response = requests.get(f"{SOLR_URL}/select", params=params)
        response.raise_for_status()
        results = response.json()

        if results['response']['numFound'] > 0:
            doc = results['response']['docs'][0]

            # Get related documents using parent_id from schema
            related_docs = []
            if 'parent_id' in doc and doc.get('parent_id'):
                parent_id = doc["parent_id"]
                if isinstance(parent_id, str):  # Check if parent_id is a string first
                    if parent_id.startswith('t1_') or parent_id.startswith('t3_'):
                        parent_id = parent_id.replace("t1_", "").replace("t3_", "")

                    parent_params = {
                        'q': f'id:"{parent_id}"',
                        'fl': '*',
                        'wt': 'json'
                    }

                    parent_response = requests.get(f"{SOLR_URL}/select", params=parent_params)
                    parent_response.raise_for_status()
                    parent_results = parent_response.json()

                    if parent_results['response']['numFound'] > 0:
                        related_docs.append(parent_results['response']['docs'][0])

            # Process keywords and entities properly
            keywords = []
            entities = []

            # Handle keywords field - safely process regardless of type
            if 'keywords' in doc:
                keywords_data = doc['keywords']
                if isinstance(keywords_data, list):
                    keywords = keywords_data
                elif isinstance(keywords_data, str):
                    try:
                        if keywords_data.startswith('['):
                            keywords = json.loads(keywords_data.replace("'", '"'))
                        else:
                            keywords = [keywords_data]
                    except:
                        keywords = []

            # Handle entities field - safely process regardless of type
            if 'entities' in doc:
                entities_data = doc['entities']
                if isinstance(entities_data, list):
                    entities = entities_data
                elif isinstance(entities_data, str):
                    try:
                        if entities_data.startswith('['):
                            entities = json.loads(entities_data.replace("'", '"'))
                        else:
                            entities = [entities_data]
                    except:
                        entities = []

            return render_template('document.html', doc=doc, related_docs=related_docs,
                                   keywords=keywords, entities=entities)
        else:
            return render_template('error.html', message="Document not found"), 404

    except Exception as e:
        error_message = f"Error retrieving document: {str(e)}"
        print(error_message)
        return render_template('error.html', message=error_message), 500


@app.route('/debug/solr')
def debug_solr():
    """Test Solr connectivity and return diagnostic information"""
    results = {
        "solr_url": SOLR_URL,
        "status": "unknown",
        "ping": None,
        "schema": None,
        "record_count": None,
        "error": None
    }

    try:
        # Test basic connectivity with ping
        ping_response = requests.get(f"{SOLR_URL}/admin/ping", timeout=5)
        results["ping"] = {
            "status_code": ping_response.status_code,
            "content": ping_response.text[:200] if ping_response.text else None
        }

        # Get schema information
        schema_response = requests.get(f"{SOLR_URL}/schema", timeout=5)
        schema_data = schema_response.json()
        results["schema"] = {
            "status_code": schema_response.status_code,
            "field_count": len(schema_data.get("schema", {}).get("fields", [])),
            "field_names": [f["name"] for f in schema_data.get("schema", {}).get("fields", [])[:10]]
        }

        # Count total records
        count_response = requests.get(f"{SOLR_URL}/select?q=*:*&rows=0&wt=json", timeout=5)
        count_data = count_response.json()
        results["record_count"] = count_data.get("response", {}).get("numFound", 0)

        results["status"] = "ok"
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)

    return jsonify(results)

def generate_visualizations(facets, query):
    """Generate visualizations from facet data"""
    visualizations = {}
    facet_fields = facets.get('facet_fields', {})

    # Platform distribution pie chart
    if 'platform' in facet_fields:
        platform_data = []
        platform_facets = facet_fields['platform']

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

    
    if 'type' in facet_fields:
        type_data = []
        type_facets = facet_fields['type']

        for i in range(0, len(type_facets), 2):
            if i + 1 < len(type_facets) and type_facets[i + 1] > 0:
                type_data.append({
                    'type': type_facets[i],
                    'count': type_facets[i + 1]
                })

        if type_data:
            df = pd.DataFrame(type_data)
            fig = px.pie(df, values='count', names='type',
                         title=f'Type Distribution for query: {query}')
            visualizations['type_pie'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    if 'sentiment' in facet_fields:
        sentiment_data = []
        sentiment_facets = facet_fields['sentiment']

        for i in range(0, len(sentiment_facets), 2):
            if i + 1 < len(sentiment_facets) and sentiment_facets[i + 1] > 0:
                sentiment_data.append({
                    'sentiment':sentiment_facets[i],
                    'count':sentiment_facets[i + 1]
                })

        if sentiment_data:
            df = pd.DataFrame(sentiment_data)
            fig = px.bar(df, x = 'sentiment', y='count',
                         title=f'Sentiment Distribution for query: {query}')
            visualizations['sentiment_bar'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        if 'created_at' in facets.get('facet_ranges', {}):
            time_data = []
            time_facets = facets['facet_ranges']['created_at']['counts']
            for i in range(0, len(time_facets), 2):
                if i + 1 < len(time_facets) and time_facets[i + 1] > 0:
                    time_data.append({
                        'date': time_facets[i],
                        'count': time_facets[i + 1]
                    })
            if time_data:
                df = pd.DataFrame(time_data)
                fig = px.line(df, x='date', y='count',
                            title=f'Timeline for query: {query}')
                visualizations['time_series'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return visualizations


@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message="Server error: " + str(e)), 500


if __name__ == '__main__':
    # Check if Solr is accessible before starting the app
    try:
        test_solr_connection()
        test_response = requests.get(f"{SOLR_URL}/admin/ping")
        test_response.raise_for_status()
        print("Successfully connected to Solr")
    except Exception as e:
        print(f"WARNING: Could not connect to Solr: {e}")
        print("The application may not function correctly without Solr")

    app.run(debug=True, port=5000)