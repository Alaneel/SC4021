# advanced_features/timeline_search.py
"""Timeline search and trend analysis for streaming opinions."""

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class TimelineSearch:
    """Timeline search and trend analysis for streaming opinions."""

    def __init__(self, solr_url="http://localhost:8983/solr/streaming_opinions"):
        """
        Initialize the timeline search.

        Args:
            solr_url (str): URL of the Solr instance
        """
        self.solr_url = solr_url

    def get_time_range(self):
        """
        Get the min and max dates in the dataset.

        Returns:
            tuple: (min_date, max_date) as datetime objects
        """
        # Query for earliest date
        params = {
            'q': '*:*',
            'sort': 'created_at asc',
            'fl': 'created_at',
            'rows': 1,
            'wt': 'json'
        }

        try:
            response = requests.get(f"{self.solr_url}/select", params=params)
            response.raise_for_status()
            results = response.json()

            if results['response']['numFound'] == 0:
                return None, None

            min_date_str = results['response']['docs'][0].get('created_at')
            min_date = datetime.fromisoformat(min_date_str.replace('Z', '+00:00'))

            # Query for latest date
            params['sort'] = 'created_at desc'

            response = requests.get(f"{self.solr_url}/select", params=params)
            response.raise_for_status()
            results = response.json()

            max_date_str = results['response']['docs'][0].get('created_at')
            max_date = datetime.fromisoformat(max_date_str.replace('Z', '+00:00'))

            return min_date, max_date
        except Exception as e:
            print(f"Error getting time range: {e}")
            return None, None

    def search_by_timeframe(self, query, start_date, end_date, filters=None, rows=10):
        """
        Search for documents within a specific timeframe.

        Args:
            query (str): Search query
            start_date (str): ISO format start date (YYYY-MM-DD)
            end_date (str): ISO format end date (YYYY-MM-DD)
            filters (list): Additional filter queries
            rows (int): Number of results to return

        Returns:
            dict: Search results
        """
        # Format dates for Solr
        date_filter = f"created_at:[{start_date}T00:00:00Z TO {end_date}T23:59:59Z]"

        # Prepare filter queries
        fq = [date_filter]
        if filters:
            fq.extend(filters)

        # Prepare search params
        params = {
            'q': query,
            'fq': fq,
            'rows': rows,
            'fl': 'id,text,title,platform,sentiment,created_at,score',
            'sort': 'created_at desc',
            'wt': 'json'
        }

        try:
            response = requests.get(f"{self.solr_url}/select", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error searching by timeframe: {e}")
            return {'response': {'docs': [], 'numFound': 0}}

    def get_time_distribution(self, query, interval='month', filters=None):
        """
        Get time distribution of documents matching a query.

        Args:
            query (str): Search query
            interval (str): Time interval (day, week, month, year)
            filters (list): Additional filter queries

        Returns:
            dict: Time distribution data
        """
        # Map interval to Solr gap
        gap_map = {
            'day': '+1DAY',
            'week': '+7DAYS',
            'month': '+1MONTH',
            'year': '+1YEAR'
        }

        gap = gap_map.get(interval, '+1MONTH')

        # Get min and max dates
        min_date, max_date = self.get_time_range()

        if not min_date or not max_date:
            return {'counts': [], 'timeline': []}

        # Format dates for Solr
        start_date = min_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date = max_date.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Prepare params
        params = {
            'q': query,
            'rows': 0,
            'facet': 'true',
            'facet.range': 'created_at',
            'facet.range.start': start_date,
            'facet.range.end': end_date,
            'facet.range.gap': gap,
            'wt': 'json'
        }

        # Add filters if provided
        if filters:
            params['fq'] = filters

        try:
            response = requests.get(f"{self.solr_url}/select", params=params)
            response.raise_for_status()
            results = response.json()

            # Extract facet counts
            facet_counts = results.get('facet_counts', {}).get('facet_ranges', {}).get('created_at', {})

            # Prepare time distribution data
            time_data = {
                'counts': facet_counts.get('counts', []),
                'timeline': []
            }

            # Convert to structured timeline data
            dates = facet_counts.get('counts', [])[::2]  # Get dates (every other item)
            counts = facet_counts.get('counts', [])[1::2]  # Get counts (every other item starting from index 1)

            for date, count in zip(dates, counts):
                time_data['timeline'].append({
                    'date': date,
                    'count': count
                })

            return time_data
        except Exception as e:
            print(f"Error getting time distribution: {e}")
            return {'counts': [], 'timeline': []}

    def get_sentiment_over_time(self, query, interval='month', filters=None):
        """
        Get sentiment distribution over time for documents matching a query.

        Args:
            query (str): Search query
            interval (str): Time interval (day, week, month, year)
            filters (list): Additional filter queries

        Returns:
            dict: Sentiment over time data
        """
        # Map interval to Solr gap
        gap_map = {
            'day': '+1DAY',
            'week': '+7DAYS',
            'month': '+1MONTH',
            'year': '+1YEAR'
        }

        gap = gap_map.get(interval, '+1MONTH')

        # Get min and max dates
        min_date, max_date = self.get_time_range()

        if not min_date or not max_date:
            return {'dates': [], 'positive': [], 'negative': [], 'neutral': []}

        # Format dates for Solr
        start_date = min_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_date = max_date.strftime('%Y-%m-%dT%H:%M:%SZ')

        # We need to make separate queries for each sentiment
        sentiments = ['positive', 'negative', 'neutral']
        sentiment_data = {sentiment: [] for sentiment in sentiments}
        sentiment_data['dates'] = []

        for sentiment in sentiments:
            # Prepare params
            params = {
                'q': query,
                'fq': [f'sentiment:{sentiment}'],
                'rows': 0,
                'facet': 'true',
                'facet.range': 'created_at',
                'facet.range.start': start_date,
                'facet.range.end': end_date,
                'facet.range.gap': gap,
                'wt': 'json'
            }

            # Add additional filters if provided
            if filters:
                if isinstance(filters, list):
                    params['fq'].extend(filters)
                else:
                    params['fq'].append(filters)

            try:
                response = requests.get(f"{self.solr_url}/select", params=params)
                response.raise_for_status()
                results = response.json()

                # Extract facet counts
                facet_counts = results.get('facet_counts', {}).get('facet_ranges', {}).get('created_at', {})

                # Save dates if first sentiment
                if sentiment == sentiments[0]:
                    dates = facet_counts.get('counts', [])[::2]  # Get dates (every other item)
                    sentiment_data['dates'] = dates

                # Extract counts
                counts = facet_counts.get('counts', [])[1::2]  # Get counts (every other item starting from index 1)
                sentiment_data[sentiment] = counts
            except Exception as e:
                print(f"Error getting {sentiment} sentiment over time: {e}")
                sentiment_data[sentiment] = [0] * len(sentiment_data['dates'])

        return sentiment_data

    def get_platform_activity_over_time(self, interval='month', filters=None):
        """
        Get activity by platform over time.

        Args:
            interval (str): Time interval (day, week, month, year)
            filters (list): Additional filter queries

        Returns:
            dict: Platform activity over time data
        """
        # Get platforms
        params = {
            'q': '*:*',
            'rows': 0,
            'facet': 'true',
            'facet.field': 'platform',
            'facet.mincount': 1,
            'wt': 'json'
        }

        try:
            response = requests.get(f"{self.solr_url}/select", params=params)
            response.raise_for_status()
            results = response.json()

            # Extract platforms
            facet_fields = results.get('facet_counts', {}).get('facet_fields', {})
            platform_facet = facet_fields.get('platform', [])

            # Convert to list of platforms (every other item is a platform name)
            platforms = platform_facet[::2]

            # Now get time data for each platform
            platform_data = {'dates': []}

            for platform in platforms:
                time_data = self.get_time_distribution(
                    f"platform:{platform}",
                    interval=interval,
                    filters=filters
                )

                if platform_data['dates'] == [] and time_data['timeline']:
                    # Initialize dates
                    platform_data['dates'] = [item['date'] for item in time_data['timeline']]

                # Add platform counts
                platform_data[platform] = [item['count'] for item in time_data['timeline']]

            return platform_data
        except Exception as e:
            print(f"Error getting platform activity: {e}")
            return {'dates': []}

    def get_topic_trend_data(self, topics, interval='month', filters=None):
        """
        Get trends for specific topics over time.

        Args:
            topics (dict): Dictionary mapping topic names to query terms
            interval (str): Time interval (day, week, month, year)
            filters (list): Additional filter queries

        Returns:
            dict: Topic trend data over time
        """
        topic_data = {'dates': []}

        # Get time data for each topic
        for topic_name, query in topics.items():
            time_data = self.get_time_distribution(
                query,
                interval=interval,
                filters=filters
            )

            if topic_data['dates'] == [] and time_data['timeline']:
                # Initialize dates
                topic_data['dates'] = [item['date'] for item in time_data['timeline']]

            # Add topic counts
            topic_data[topic_name] = [item['count'] for item in time_data['timeline']]

        return topic_data

    def create_timeline_visualization(self, data, title="Document Count Over Time",
                                      x_label="Date", y_label="Count", as_html=True):
        """
        Create a timeline visualization.

        Args:
            data (dict): Timeline data
            title (str): Chart title
            x_label (str): X-axis label
            y_label (str): Y-axis label
            as_html (bool): Whether to return HTML

        Returns:
            str or Plotly figure: Visualization
        """
        if 'timeline' in data and data['timeline']:
            # Extract dates and counts
            dates = [item['date'] for item in data['timeline']]
            counts = [item['count'] for item in data['timeline']]

            # Create plotly figure
            fig = px.line(
                x=dates,
                y=counts,
                title=title,
                labels={'x': x_label, 'y': y_label}
            )

            # Add markers for data points
            fig.update_traces(mode='lines+markers')

            if as_html:
                return fig.to_html()
            else:
                return fig
        else:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                showarrow=False,
                font=dict(size=20)
            )

            if as_html:
                return fig.to_html()
            else:
                return fig

    def create_sentiment_timeline_visualization(self, data, title="Sentiment Over Time",
                                                x_label="Date", y_label="Count", as_html=True):
        """
        Create a sentiment timeline visualization.

        Args:
            data (dict): Sentiment timeline data
            title (str): Chart title
            x_label (str): X-axis label
            y_label (str): Y-axis label
            as_html (bool): Whether to return HTML

        Returns:
            str or Plotly figure: Visualization
        """
        if 'dates' in data and data['dates']:
            # Create plotly figure
            fig = go.Figure()

            # Add line for each sentiment
            if 'positive' in data:
                fig.add_trace(go.Scatter(
                    x=data['dates'],
                    y=data['positive'],
                    mode='lines+markers',
                    name='Positive',
                    line=dict(color='green', width=2)
                ))

            if 'negative' in data:
                fig.add_trace(go.Scatter(
                    x=data['dates'],
                    y=data['negative'],
                    mode='lines+markers',
                    name='Negative',
                    line=dict(color='red', width=2)
                ))

            if 'neutral' in data:
                fig.add_trace(go.Scatter(
                    x=data['dates'],
                    y=data['neutral'],
                    mode='lines+markers',
                    name='Neutral',
                    line=dict(color='gray', width=2)
                ))

            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                legend_title="Sentiment",
                hovermode="x unified"
            )

            if as_html:
                return fig.to_html()
            else:
                return fig
        else:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                showarrow=False,
                font=dict(size=20)
            )

            if as_html:
                return fig.to_html()
            else:
                return fig

    def create_platform_timeline_visualization(self, data, title="Platform Activity Over Time",
                                               x_label="Date", y_label="Count", as_html=True):
        """
        Create a platform timeline visualization.

        Args:
            data (dict): Platform timeline data
            title (str): Chart title
            x_label (str): X-axis label
            y_label (str): Y-axis label
            as_html (bool): Whether to return HTML

        Returns:
            str or Plotly figure: Visualization
        """
        if 'dates' in data and data['dates']:
            # Get platforms
            platforms = [key for key in data.keys() if key != 'dates']

            if not platforms:
                # Return empty figure
                fig = go.Figure()
                fig.add_annotation(
                    text="No platform data available",
                    showarrow=False,
                    font=dict(size=20)
                )

                if as_html:
                    return fig.to_html()
                else:
                    return fig

            # Create plotly figure
            fig = go.Figure()

            # Platform colors
            platform_colors = {
                'netflix': '#E50914',
                'disney+': '#113CCF',
                'hulu': '#1CE783',
                'amazon prime': '#00A8E1',
                'hbo max': '#5822B4',
                'apple tv+': '#000000',
                'peacock': '#FFFFFF',
                'paramount+': '#0064FF'
            }

            # Add line for each platform
            for platform in platforms:
                color = platform_colors.get(platform.lower(), '#6c757d')

                fig.add_trace(go.Scatter(
                    x=data['dates'],
                    y=data[platform],
                    mode='lines+markers',
                    name=platform,
                    line=dict(color=color, width=2)
                ))

            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                legend_title="Platform",
                hovermode="x unified"
            )

            if as_html:
                return fig.to_html()
            else:
                return fig
        else:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                showarrow=False,
                font=dict(size=20)
            )

            if as_html:
                return fig.to_html()
            else:
                return fig

    def create_topic_trends_visualization(self, data, title="Topic Trends Over Time",
                                          x_label="Date", y_label="Count", as_html=True):
        """
        Create a topic trends visualization.

        Args:
            data (dict): Topic trends data
            title (str): Chart title
            x_label (str): X-axis label
            y_label (str): Y-axis label
            as_html (bool): Whether to return HTML

        Returns:
            str or Plotly figure: Visualization
        """
        if 'dates' in data and data['dates']:
            # Get topics
            topics = [key for key in data.keys() if key != 'dates']

            if not topics:
                # Return empty figure
                fig = go.Figure()
                fig.add_annotation(
                    text="No topic data available",
                    showarrow=False,
                    font=dict(size=20)
                )

                if as_html:
                    return fig.to_html()
                else:
                    return fig

            # Create plotly figure
            fig = go.Figure()

            # Add line for each topic
            for topic in topics:
                fig.add_trace(go.Scatter(
                    x=data['dates'],
                    y=data[topic],
                    mode='lines+markers',
                    name=topic
                ))

            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                legend_title="Topic",
                hovermode="x unified"
            )

            if as_html:
                return fig.to_html()
            else:
                return fig
        else:
            # Return empty figure
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                showarrow=False,
                font=dict(size=20)
            )

            if as_html:
                return fig.to_html()
            else:
                return fig

    def create_interactive_dashboard(self, query="*:*", interval="month", filters=None,
                                     topics=None, as_html=True):
        """
        Create an interactive timeline dashboard.

        Args:
            query (str): Main query string
            interval (str): Time interval
            filters (list): Filter queries
            topics (dict): Dictionary of topics to track
            as_html (bool): Whether to return HTML

        Returns:
            str or Plotly figure: Dashboard
        """
        # Default topics if none provided
        if topics is None:
            topics = {
                "Price Increase": "price increase OR subscription cost OR expensive",
                "Content Quality": "content quality OR shows OR movies OR original",
                "Technical Issues": "technical OR buffering OR streaming quality OR crash",
                "UI/UX": "interface OR app OR user experience OR design"
            }

        # Get timeline data
        timeline_data = self.get_time_distribution(query, interval=interval, filters=filters)

        # Get sentiment data
        sentiment_data = self.get_sentiment_over_time(query, interval=interval, filters=filters)

        # Get platform data
        platform_data = self.get_platform_activity_over_time(interval=interval, filters=filters)

        # Get topic trends
        topic_data = self.get_topic_trend_data(topics, interval=interval, filters=filters)

        # Create dashboard
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Document Count Over Time",
                "Sentiment Over Time",
                "Platform Activity",
                "Topic Trends"
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )

        # Add timeline chart (top left)
        if 'timeline' in timeline_data and timeline_data['timeline']:
            dates = [item['date'] for item in timeline_data['timeline']]
            counts = [item['count'] for item in timeline_data['timeline']]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=counts,
                    mode='lines+markers',
                    name='Document Count',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )

        # Add sentiment chart (top right)
        if 'dates' in sentiment_data and sentiment_data['dates']:
            # Add positive sentiment
            if 'positive' in sentiment_data:
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_data['dates'],
                        y=sentiment_data['positive'],
                        mode='lines+markers',
                        name='Positive',
                        line=dict(color='green', width=2)
                    ),
                    row=1, col=2
                )

            # Add negative sentiment
            if 'negative' in sentiment_data:
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_data['dates'],
                        y=sentiment_data['negative'],
                        mode='lines+markers',
                        name='Negative',
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=2
                )

            # Add neutral sentiment
            if 'neutral' in sentiment_data:
                fig.add_trace(
                    go.Scatter(
                        x=sentiment_data['dates'],
                        y=sentiment_data['neutral'],
                        mode='lines+markers',
                        name='Neutral',
                        line=dict(color='gray', width=2)
                    ),
                    row=1, col=2
                )

        # Add platform chart (bottom left)
        if 'dates' in platform_data and platform_data['dates']:
            # Platform colors
            platform_colors = {
                'netflix': '#E50914',
                'disney+': '#113CCF',
                'hulu': '#1CE783',
                'amazon prime': '#00A8E1',
                'hbo max': '#5822B4',
                'apple tv+': '#000000',
                'peacock': '#FFFFFF',
                'paramount+': '#0064FF'
            }

            # Add line for each platform
            for platform in [key for key in platform_data.keys() if key != 'dates']:
                color = platform_colors.get(platform.lower(), '#6c757d')

                fig.add_trace(
                    go.Scatter(
                        x=platform_data['dates'],
                        y=platform_data[platform],
                        mode='lines+markers',
                        name=platform,
                        line=dict(color=color, width=2)
                    ),
                    row=2, col=1
                )

        # Add topic chart (bottom right)
        if 'dates' in topic_data and topic_data['dates']:
            for topic in [key for key in topic_data.keys() if key != 'dates']:
                fig.add_trace(
                    go.Scatter(
                        x=topic_data['dates'],
                        y=topic_data[topic],
                        mode='lines+markers',
                        name=topic
                    ),
                    row=2, col=2
                )

        # Update layout
        fig.update_layout(
            title=f"Streaming Opinions Timeline Dashboard - {query}",
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )

        # Update axes titles
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)

        if as_html:
            return fig.to_html()
        else:
            return fig

    def get_peak_periods(self, query, interval='month', top_n=3, filters=None):
        """
        Identify peak periods for a given query.

        Args:
            query (str): Search query
            interval (str): Time interval (day, week, month, year)
            top_n (int): Number of peak periods to return
            filters (list): Additional filter queries

        Returns:
            list: Peak periods with their document counts
        """
        # Get time distribution
        time_data = self.get_time_distribution(query, interval=interval, filters=filters)

        if not time_data['timeline']:
            return []

        # Convert to dataframe for easier analysis
        df = pd.DataFrame(time_data['timeline'])

        # Sort by count
        df = df.sort_values('count', ascending=False)

        # Get top N periods
        top_periods = df.head(top_n).to_dict('records')

        return top_periods

    def get_sentiment_shift_periods(self, query, interval='month', threshold=0.2, filters=None):
        """
        Identify periods with significant sentiment shifts.

        Args:
            query (str): Search query
            interval (str): Time interval (day, week, month, year)
            threshold (float): Minimum shift to be considered significant
            filters (list): Additional filter queries

        Returns:
            list: Periods with significant sentiment shifts
        """
        # Get sentiment over time
        sentiment_data = self.get_sentiment_over_time(query, interval=interval, filters=filters)

        if not sentiment_data['dates']:
            return []

        # Create dataframe
        df = pd.DataFrame({
            'date': sentiment_data['dates'],
            'positive': sentiment_data.get('positive', [0] * len(sentiment_data['dates'])),
            'negative': sentiment_data.get('negative', [0] * len(sentiment_data['dates'])),
            'neutral': sentiment_data.get('neutral', [0] * len(sentiment_data['dates']))
        })

        # Calculate total counts
        df['total'] = df['positive'] + df['negative'] + df['neutral']

        # Calculate sentiment ratios
        for sentiment in ['positive', 'negative', 'neutral']:
            df[f'{sentiment}_ratio'] = df[sentiment] / df['total'].replace(0, 1)

        # Calculate shifts
        shifts = []

        for i in range(1, len(df)):
            pos_shift = df['positive_ratio'].iloc[i] - df['positive_ratio'].iloc[i - 1]
            neg_shift = df['negative_ratio'].iloc[i] - df['negative_ratio'].iloc[i - 1]

            # If either shift is significant
            if abs(pos_shift) >= threshold or abs(neg_shift) >= threshold:
                shifts.append({
                    'date': df['date'].iloc[i],
                    'prev_date': df['date'].iloc[i - 1],
                    'pos_shift': pos_shift,
                    'neg_shift': neg_shift,
                    'total_docs': df['total'].iloc[i]
                })

        return shifts


# Example Flask integration
def integrate_with_flask(app, solr_url):
    """
    Integrate timeline search with Flask app.

    Args:
        app: Flask application
        solr_url (str): URL of the Solr instance
    """
    from flask import request, jsonify, render_template

    timeline_search = TimelineSearch(solr_url)

    @app.route('/timeline')
    def timeline_page():
        """Render timeline search page."""
        return render_template('timeline.html')

    @app.route('/api/timeline/search')
    def timeline_search_api():
        """API endpoint for timeline search."""
        query = request.args.get('q', '*:*')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        filters = request.args.getlist('fq')

        if not start_date or not end_date:
            # Get min and max dates if not provided
            min_date, max_date = timeline_search.get_time_range()

            if not min_date or not max_date:
                return jsonify({'error': 'No date range available'}), 400

            start_date = start_date or min_date.strftime('%Y-%m-%d')
            end_date = end_date or max_date.strftime('%Y-%m-%d')

        results = timeline_search.search_by_timeframe(
            query=query,
            start_date=start_date,
            end_date=end_date,
            filters=filters
        )

        return jsonify(results)

    @app.route('/api/timeline/distribution')
    def timeline_distribution_api():
        """API endpoint for time distribution data."""
        query = request.args.get('q', '*:*')
        interval = request.args.get('interval', 'month')
        filters = request.args.getlist('fq')

        time_data = timeline_search.get_time_distribution(
            query=query,
            interval=interval,
            filters=filters
        )

        return jsonify(time_data)

    @app.route('/api/timeline/sentiment')
    def timeline_sentiment_api():
        """API endpoint for sentiment over time data."""
        query = request.args.get('q', '*:*')
        interval = request.args.get('interval', 'month')
        filters = request.args.getlist('fq')

        sentiment_data = timeline_search.get_sentiment_over_time(
            query=query,
            interval=interval,
            filters=filters
        )

        return jsonify(sentiment_data)

    @app.route('/api/timeline/platforms')
    def timeline_platforms_api():
        """API endpoint for platform activity over time data."""
        interval = request.args.get('interval', 'month')
        filters = request.args.getlist('fq')

        platform_data = timeline_search.get_platform_activity_over_time(
            interval=interval,
            filters=filters
        )

        return jsonify(platform_data)

    @app.route('/api/timeline/dashboard')
    def timeline_dashboard_api():
        """API endpoint for interactive dashboard."""
        query = request.args.get('q', '*:*')
        interval = request.args.get('interval', 'month')
        filters = request.args.getlist('fq')

        html = timeline_search.create_interactive_dashboard(
            query=query,
            interval=interval,
            filters=filters
        )

        return jsonify({'html': html})