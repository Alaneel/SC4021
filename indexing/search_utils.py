"""
Search utilities for the EV opinions search engine
"""
import pysolr
import json
import logging
from datetime import datetime
import pandas as pd
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from config.app_config import SOLR_FULL_URL

logger = logging.getLogger(__name__)


def execute_search(query, filters=None, facets=None, start=0, rows=10, sort=None):
    """
    Execute a search against the Solr index

    Args:
        query (str): The search query
        filters (list): A list of filter queries
        facets (list): Facet fields to include
        start (int): Starting offset for results
        rows (int): Number of rows to return
        sort (str): Sort order

    Returns:
        dict: Search results
    """
    solr = pysolr.Solr(SOLR_FULL_URL)

    # Default parameters
    search_params = {
        'start': start,
        'rows': rows,
        'hl': 'on',
        'hl.fl': 'text,title',
        'hl.snippets': 3,
        'hl.fragsize': 150,
    }

    # Add filters if provided
    if filters:
        search_params['fq'] = filters

    # Add sort if provided
    if sort:
        search_params['sort'] = sort

    # Add faceting if requested
    if facets:
        search_params['facet'] = 'on'
        search_params['facet.field'] = facets

        # Always include date range facet
        search_params['facet.range'] = 'created_utc'
        search_params['facet.range.start'] = 'NOW/YEAR-2YEAR'
        search_params['facet.range.end'] = 'NOW'
        search_params['facet.range.gap'] = '+1MONTH'

    # Execute the search
    try:
        logger.info(f"Executing search: {query}")
        logger.debug(f"Search params: {search_params}")

        search_results = solr.search(query, **search_params)

        # Format results
        results = {
            'query': query,
            'num_found': search_results.hits,
            'start': start,
            'rows': rows,
            'docs': list(search_results),
            'highlighting': getattr(search_results, 'highlighting', {}),
        }

        # Add facets if present
        if facets and hasattr(search_results, 'facets'):
            results['facets'] = search_results.facets

        logger.info(f"Search returned {search_results.hits} results")
        return results

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return {
            'error': str(e),
            'query': query,
            'num_found': 0,
            'docs': []
        }


def format_search_results(results, include_visualizations=True):
    """
    Format search results for the UI, including generating visualizations

    Args:
        results (dict): Search results from execute_search
        include_visualizations (bool): Whether to include data visualizations

    Returns:
        dict: Formatted results with visualizations
    """
    formatted = results.copy()

    # Generate visualizations if requested and we have results
    if include_visualizations and results['num_found'] > 0 and 'facets' in results:
        # Generate sentiment distribution chart
        if ('facet_fields' in results['facets'] and
                'sentiment' in results['facets']['facet_fields']):
            sentiment_data = dict(zip(
                results['facets']['facet_fields']['sentiment'][::2],
                results['facets']['facet_fields']['sentiment'][1::2]
            ))
            formatted['sentiment_chart'] = generate_sentiment_chart(sentiment_data)

        # Generate time series chart
        if ('facet_ranges' in results['facets'] and
                'created_utc' in results['facets']['facet_ranges']):
            time_data = results['facets']['facet_ranges']['created_utc']['counts']
            time_dict = dict(zip(time_data[::2], time_data[1::2]))
            formatted['time_chart'] = generate_time_chart(time_dict)

        # Generate word cloud
        if len(results['docs']) > 0:
            formatted['wordcloud'] = generate_wordcloud(
                [doc.get('text', '') for doc in results['docs'] if 'text' in doc]
            )

    return formatted


def generate_sentiment_chart(sentiment_data):
    """
    Generate a pie chart of sentiment distribution

    Args:
        sentiment_data (dict): Sentiment distribution data

    Returns:
        str: Base64-encoded PNG image
    """
    # Check if we have data
    if not sentiment_data:
        return None

    # Map sentiment labels to readable format and colors
    label_map = {
        'positive': 'Positive',
        'negative': 'Negative',
        'neutral': 'Neutral'
    }

    colors = {
        'positive': '#4CAF50',  # Green
        'negative': '#F44336',  # Red
        'neutral': '#9E9E9E'  # Gray
    }

    # Prepare data
    labels = [label_map.get(k, k) for k in sentiment_data.keys()]
    sizes = list(sentiment_data.values())
    chart_colors = [colors.get(k, '#2196F3') for k in sentiment_data.keys()]

    # Create the pie chart
    plt.figure(figsize=(7, 5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=chart_colors)
    plt.axis('equal')
    plt.title('Sentiment Distribution')

    # Save to base64 for embedding in HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()

    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    return base64.b64encode(image_png).decode('utf-8')


def generate_time_chart(time_data):
    """
    Generate a time series chart of opinions over time

    Args:
        time_data (dict): Time series data

    Returns:
        str: Base64-encoded PNG image
    """
    # Check if we have data
    if not time_data:
        return None

    # Convert string dates to datetime objects
    dates = [datetime.strptime(d, '%Y-%m-%dT%H:%M:%SZ') for d in time_data.keys()]
    counts = list(time_data.values())

    # Create the time series chart
    plt.figure(figsize=(10, 4))
    plt.plot(dates, counts, marker='o', linestyle='-', color='#2196F3')
    plt.xlabel('Date')
    plt.ylabel('Number of Opinions')
    plt.title('Opinion Timeline')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Format x-axis to show dates nicely
    plt.gcf().autofmt_xdate()

    # Save to base64 for embedding in HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()

    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    return base64.b64encode(image_png).decode('utf-8')


def generate_wordcloud(texts):
    """
    Generate a word cloud from the search results

    Args:
        texts (list): List of text content

    Returns:
        str: Base64-encoded PNG image
    """
    # Check if we have data
    if not texts:
        return None

    # Combine all texts
    combined_text = ' '.join(texts)

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100,
        colormap='viridis',
        contour_width=1,
        contour_color='steelblue'
    ).generate(combined_text)

    # Create figure
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Key Terms')
    plt.tight_layout(pad=0)

    # Save to base64 for embedding in HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()

    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    return base64.b64encode(image_png).decode('utf-8')