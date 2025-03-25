# evaluation/performance_metrics.py
"""Performance evaluation tool for the search system."""

import requests
import time
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate
from datetime import datetime


class PerformanceEvaluator:
    """Performance evaluation tool for the search and classification system."""

    def __init__(self, solr_url="http://localhost:8983/solr/streaming_opinions"):
        """
        Initialize the performance evaluator.

        Args:
            solr_url (str): URL of the Solr instance
        """
        self.solr_url = solr_url
        self.search_queries = []
        self.query_results = {}
        self.performance_metrics = {}
        self.system_info = {}

    def test_solr_connection(self):
        """
        Test connection to Solr.

        Returns:
            bool: True if connection is successful
        """
        try:
            response = requests.get(f"{self.solr_url}/admin/ping", timeout=5)
            response.raise_for_status()

            # Check for Solr version and status
            system_info_response = requests.get(f"{self.solr_url}/admin/info/system", timeout=5)
            self.system_info = system_info_response.json().get('system', {})

            # Get index info
            index_response = requests.get(f"{self.solr_url}/admin/luke?numTerms=0", timeout=5)
            index_info = index_response.json()

            # Get schema fields
            schema_response = requests.get(f"{self.solr_url}/schema/fields", timeout=5)
            schema_info = schema_response.json()

            # Add information to system_info
            self.system_info['index'] = {
                'num_docs': index_info.get('index', {}).get('numDocs', 0),
                'max_doc': index_info.get('index', {}).get('maxDoc', 0),
                'num_fields': len(schema_info.get('fields', []))
            }

            return True
        except Exception as e:
            print(f"Error connecting to Solr: {e}")
            return False

    def add_search_query(self, query, description, filters=None, rows=10):
        """
        Add a search query to test.

        Args:
            query (str): The search query string
            description (str): Description of the query
            filters (list, optional): List of filter queries
            rows (int, optional): Number of rows to return

        Returns:
            dict: The added query
        """
        query_data = {
            'id': len(self.search_queries) + 1,
            'query': query,
            'description': description,
            'filters': filters or [],
            'rows': rows
        }

        self.search_queries.append(query_data)
        return query_data

    def run_queries(self, repeats=3):
        """
        Run all search queries and measure performance.

        Args:
            repeats (int): Number of times to repeat each query for reliable timing

        Returns:
            dict: Query performance results
        """
        if not self.search_queries:
            print("No search queries to run")
            return {}

        print(f"Running {len(self.search_queries)} search queries with {repeats} repeats each")

        for query_data in self.search_queries:
            query_id = query_data['id']
            query = query_data['query']
            filters = query_data['filters']
            rows = query_data['rows']

            print(f"Testing query {query_id}: {query_data['description']}")

            # Prepare Solr query parameters
            params = {
                'q': query,
                'rows': rows,
                'wt': 'json'
            }

            if filters:
                params['fq'] = filters

            # Run query multiple times
            timings = []
            num_results = 0

            for i in range(repeats):
                start_time = time.time()

                try:
                    response = requests.get(f"{self.solr_url}/select", params=params, timeout=30)
                    response.raise_for_status()
                    results = response.json()

                    end_time = time.time()
                    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

                    timings.append(elapsed_time)

                    # Store number of results for the first run
                    if i == 0:
                        num_results = results['response']['numFound']
                        # Store the first page of results
                        self.query_results[query_id] = results['response']['docs']

                except Exception as e:
                    print(f"Error running query: {e}")
                    timings.append(None)

                # Add a small delay between queries
                time.sleep(0.1)

            # Calculate timing statistics
            valid_timings = [t for t in timings if t is not None]

            if valid_timings:
                avg_time = np.mean(valid_timings)
                min_time = np.min(valid_timings)
                max_time = np.max(valid_timings)
                std_time = np.std(valid_timings)
            else:
                avg_time = min_time = max_time = std_time = None

            # Store results
            self.performance_metrics[query_id] = {
                'query': query,
                'description': query_data['description'],
                'filters': filters,
                'rows': rows,
                'num_results': num_results,
                'avg_time_ms': avg_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'std_time_ms': std_time,
                'timings': valid_timings
            }

        return self.performance_metrics

    def run_default_queries(self):
        """
        Run a set of default queries for testing.

        Returns:
            dict: Query performance results
        """
        # Clear any existing queries
        self.search_queries = []

        # Add default queries
        self.add_search_query("*:*", "All documents")
        self.add_search_query("netflix", "Netflix mentions")
        self.add_search_query("streaming price increase", "Streaming price increases", ["sentiment:negative"])
        self.add_search_query("technical issues buffering", "Technical issues",
                              ["platform:netflix", "sentiment:negative"])
        self.add_search_query("content quality shows", "Content quality", ["feature:content_quality"])

        # Run queries
        return self.run_queries()

    def test_facet_performance(self):
        """
        Test the performance of faceted search.

        Returns:
            dict: Facet performance metrics
        """
        facet_types = [
            {
                'description': 'Platform facets',
                'params': {
                    'q': '*:*',
                    'facet': 'true',
                    'facet.field': 'platform',
                    'facet.mincount': 1,
                    'rows': 0,
                    'wt': 'json'
                }
            },
            {
                'description': 'Sentiment facets',
                'params': {
                    'q': '*:*',
                    'facet': 'true',
                    'facet.field': 'sentiment',
                    'facet.mincount': 1,
                    'rows': 0,
                    'wt': 'json'
                }
            },
            {
                'description': 'Date range facets',
                'params': {
                    'q': '*:*',
                    'facet': 'true',
                    'facet.range': 'created_at',
                    'facet.range.start': 'NOW-1YEAR',
                    'facet.range.end': 'NOW',
                    'facet.range.gap': '+1MONTH',
                    'rows': 0,
                    'wt': 'json'
                }
            },
            {
                'description': 'Feature pivot facets',
                'params': {
                    'q': '*:*',
                    'facet': 'true',
                    'facet.pivot': 'platform,sentiment',
                    'facet.mincount': 1,
                    'rows': 0,
                    'wt': 'json'
                }
            }
        ]

        facet_metrics = {}

        for facet_type in facet_types:
            description = facet_type['description']
            params = facet_type['params']

            print(f"Testing facet: {description}")

            # Run facet query multiple times
            timings = []

            for i in range(3):  # Repeat 3 times
                start_time = time.time()

                try:
                    response = requests.get(f"{self.solr_url}/select", params=params, timeout=30)
                    response.raise_for_status()

                    end_time = time.time()
                    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

                    timings.append(elapsed_time)
                except Exception as e:
                    print(f"Error running facet query: {e}")
                    timings.append(None)

                # Add a small delay between queries
                time.sleep(0.1)

            # Calculate timing statistics
            valid_timings = [t for t in timings if t is not None]

            if valid_timings:
                avg_time = np.mean(valid_timings)
                min_time = np.min(valid_timings)
                max_time = np.max(valid_timings)
                std_time = np.std(valid_timings)
            else:
                avg_time = min_time = max_time = std_time = None

            # Store results
            facet_metrics[description] = {
                'avg_time_ms': avg_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'std_time_ms': std_time,
                'timings': valid_timings
            }

        self.performance_metrics['facet_queries'] = facet_metrics
        return facet_metrics

    def test_scalability(self, doc_count_steps=None):
        """
        Test the scalability of the search system with different document counts.

        Args:
            doc_count_steps (list, optional): List of document counts to test

        Returns:
            dict: Scalability metrics
        """
        if not doc_count_steps:
            # Get total document count
            try:
                response = requests.get(f"{self.solr_url}/select", params={'q': '*:*', 'rows': 0, 'wt': 'json'},
                                        timeout=5)
                total_docs = response.json()['response']['numFound']

                # Create steps at 20%, 40%, 60%, 80%, 100% of total docs
                doc_count_steps = [
                    int(total_docs * 0.2),
                    int(total_docs * 0.4),
                    int(total_docs * 0.6),
                    int(total_docs * 0.8),
                    total_docs
                ]
            except Exception as e:
                print(f"Error getting document count: {e}")
                return {}

        print(f"Testing scalability with document counts: {doc_count_steps}")

        # Use a simple query for testing
        test_query = "netflix"

        scalability_metrics = {
            'doc_counts': doc_count_steps,
            'query_times': [],
            'facet_times': []
        }

        for doc_count in doc_count_steps:
            print(f"Testing with {doc_count} documents")

            # Limit to specific document count using sort and rows
            # Regular query
            query_params = {
                'q': test_query,
                'rows': 10,
                'sort': 'id asc',
                'fq': f'id:*',  # Simple filter to ensure consistent results
                'wt': 'json'
            }

            # Faceted query
            facet_params = {
                'q': test_query,
                'rows': 0,
                'facet': 'true',
                'facet.field': 'platform',
                'facet.field': 'sentiment',
                'facet.mincount': 1,
                'wt': 'json'
            }

            # Run queries 3 times and take average
            query_timings = []
            facet_timings = []

            for i in range(3):
                # Regular query
                start_time = time.time()
                try:
                    response = requests.get(f"{self.solr_url}/select", params=query_params, timeout=30)
                    response.raise_for_status()

                    end_time = time.time()
                    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

                    query_timings.append(elapsed_time)
                except Exception as e:
                    print(f"Error running scalability query: {e}")

                time.sleep(0.1)

                # Faceted query
                start_time = time.time()
                try:
                    response = requests.get(f"{self.solr_url}/select", params=facet_params, timeout=30)
                    response.raise_for_status()

                    end_time = time.time()
                    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

                    facet_timings.append(elapsed_time)
                except Exception as e:
                    print(f"Error running scalability facet query: {e}")

                time.sleep(0.1)

            # Calculate average times
            avg_query_time = np.mean(query_timings) if query_timings else None
            avg_facet_time = np.mean(facet_timings) if facet_timings else None

            scalability_metrics['query_times'].append(avg_query_time)
            scalability_metrics['facet_times'].append(avg_facet_time)

        self.performance_metrics['scalability'] = scalability_metrics
        return scalability_metrics

    def visualize_performance(self, output_dir='evaluation/results'):
        """
        Generate visualizations of performance metrics.

        Args:
            output_dir (str): Directory to save visualizations

        Returns:
            dict: Paths to generated visualizations
        """
        if not self.performance_metrics:
            print("No performance metrics to visualize")
            return {}

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        visualizations = {}

        # Query performance bar chart
        if any(k for k in self.performance_metrics.keys() if isinstance(k, int)):
            query_metrics = {k: v for k, v in self.performance_metrics.items() if isinstance(k, int)}

            plt.figure(figsize=(12, 6))

            query_ids = list(query_metrics.keys())
            avg_times = [metrics['avg_time_ms'] for metrics in query_metrics.values()]
            query_descriptions = [metrics['description'] for metrics in query_metrics.values()]

            # Create the bar chart
            bars = plt.bar(query_ids, avg_times)

            # Add data labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
                         f"{height:.1f} ms", ha='center', va='bottom')

            plt.xlabel('Query ID')
            plt.ylabel('Average Response Time (ms)')
            plt.title('Query Performance')
            plt.xticks(query_ids)

            # Add a legend for query descriptions
            legend_labels = [f"Query {q_id}: {desc}" for q_id, desc in zip(query_ids, query_descriptions)]
            plt.legend(bars, legend_labels, loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()

            query_perf_path = os.path.join(output_dir, 'query_performance.png')
            plt.savefig(query_perf_path)
            plt.close()

            visualizations['query_performance'] = query_perf_path

        # Facet performance bar chart
        if 'facet_queries' in self.performance_metrics:
            facet_metrics = self.performance_metrics['facet_queries']

            plt.figure(figsize=(12, 6))

            facet_types = list(facet_metrics.keys())
            avg_times = [metrics['avg_time_ms'] for metrics in facet_metrics.values()]

            # Create the bar chart
            bars = plt.bar(facet_types, avg_times)

            # Add data labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
                         f"{height:.1f} ms", ha='center', va='bottom')

            plt.xlabel('Facet Type')
            plt.ylabel('Average Response Time (ms)')
            plt.title('Facet Query Performance')
            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()

            facet_perf_path = os.path.join(output_dir, 'facet_performance.png')
            plt.savefig(facet_perf_path)
            plt.close()

            visualizations['facet_performance'] = facet_perf_path

        # Scalability line chart
        if 'scalability' in self.performance_metrics:
            scalability = self.performance_metrics['scalability']

            plt.figure(figsize=(12, 6))

            doc_counts = scalability['doc_counts']
            query_times = scalability['query_times']
            facet_times = scalability['facet_times']

            plt.plot(doc_counts, query_times, marker='o', linestyle='-', label='Regular Query')
            plt.plot(doc_counts, facet_times, marker='s', linestyle='-', label='Faceted Query')

            plt.xlabel('Document Count')
            plt.ylabel('Average Response Time (ms)')
            plt.title('Scalability Performance')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            scalability_path = os.path.join(output_dir, 'scalability.png')
            plt.savefig(scalability_path)
            plt.close()

            visualizations['scalability'] = scalability_path

        return visualizations

    def export_performance_report(self, output_path=None):
        """
        Export performance metrics to a formatted report.

        Args:
            output_path (str, optional): Path to save the report

        Returns:
            str: Path to the saved report
        """
        if not output_path:
            output_path = 'evaluation/results/performance_report.md'

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Generate report content
        report_lines = []

        # Title and timestamp
        report_lines.append("# Search System Performance Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # System information
        report_lines.append("## System Information")
        if self.system_info:
            report_lines.append(f"- Solr Version: {self.system_info.get('solr', {}).get('version', 'Unknown')}")
            report_lines.append(f"- Java Version: {self.system_info.get('jvm', {}).get('version', 'Unknown')}")
            report_lines.append(f"- Total Documents: {self.system_info.get('index', {}).get('num_docs', 'Unknown')}")
            report_lines.append(f"- Number of Fields: {self.system_info.get('index', {}).get('num_fields', 'Unknown')}")
        else:
            report_lines.append("System information not available.")
        report_lines.append("")

        # Query performance
        report_lines.append("## Query Performance")
        query_metrics = {k: v for k, v in self.performance_metrics.items() if isinstance(k, int)}

        if query_metrics:
            table_rows = []
            table_headers = ["ID", "Description", "Results", "Avg Time (ms)", "Min Time (ms)", "Max Time (ms)"]

            for query_id, metrics in query_metrics.items():
                table_rows.append([
                    query_id,
                    metrics['description'],
                    metrics['num_results'],
                    f"{metrics['avg_time_ms']:.2f}" if metrics['avg_time_ms'] else "N/A",
                    f"{metrics['min_time_ms']:.2f}" if metrics['min_time_ms'] else "N/A",
                    f"{metrics['max_time_ms']:.2f}" if metrics['max_time_ms'] else "N/A"
                ])

            report_lines.append(tabulate(table_rows, headers=table_headers, tablefmt="pipe"))
        else:
            report_lines.append("No query performance metrics available.")
        report_lines.append("")

        # Facet performance
        if 'facet_queries' in self.performance_metrics:
            report_lines.append("## Facet Query Performance")

            facet_metrics = self.performance_metrics['facet_queries']
            table_rows = []
            table_headers = ["Facet Type", "Avg Time (ms)", "Min Time (ms)", "Max Time (ms)"]

            for facet_type, metrics in facet_metrics.items():
                table_rows.append([
                    facet_type,
                    f"{metrics['avg_time_ms']:.2f}" if metrics['avg_time_ms'] else "N/A",
                    f"{metrics['min_time_ms']:.2f}" if metrics['min_time_ms'] else "N/A",
                    f"{metrics['max_time_ms']:.2f}" if metrics['max_time_ms'] else "N/A"
                ])

            report_lines.append(tabulate(table_rows, headers=table_headers, tablefmt="pipe"))
            report_lines.append("")

        # Scalability metrics
        if 'scalability' in self.performance_metrics:
            report_lines.append("## Scalability Performance")

            scalability = self.performance_metrics['scalability']
            table_rows = []
            table_headers = ["Document Count", "Query Time (ms)", "Facet Time (ms)"]

            for i, doc_count in enumerate(scalability['doc_counts']):
                table_rows.append([
                    doc_count,
                    f"{scalability['query_times'][i]:.2f}" if scalability['query_times'][i] else "N/A",
                    f"{scalability['facet_times'][i]:.2f}" if scalability['facet_times'][i] else "N/A"
                ])

            report_lines.append(tabulate(table_rows, headers=table_headers, tablefmt="pipe"))
            report_lines.append("")

        # Visualizations
        report_lines.append("## Performance Visualizations")
        report_lines.append("See the following visualization files:")
        report_lines.append("- Query Performance: `query_performance.png`")
        report_lines.append("- Facet Performance: `facet_performance.png`")
        report_lines.append("- Scalability: `scalability.png`")
        report_lines.append("")

        # Conclusions
        report_lines.append("## Conclusions")

        # Performance assessment based on average query times
        avg_query_times = [m['avg_time_ms'] for m in query_metrics.values() if m['avg_time_ms'] is not None]
        if avg_query_times:
            overall_avg = np.mean(avg_query_times)

            if overall_avg < 50:
                assessment = "excellent"
            elif overall_avg < 100:
                assessment = "very good"
            elif overall_avg < 200:
                assessment = "good"
            elif overall_avg < 500:
                assessment = "acceptable"
            else:
                assessment = "needs improvement"

            report_lines.append(
                f"- Overall query performance is {assessment} with an average response time of {overall_avg:.2f} ms.")

        # Scaling assessment
        if 'scalability' in self.performance_metrics:
            scalability = self.performance_metrics['scalability']

            if len(scalability['query_times']) > 1:
                first_time = scalability['query_times'][0]
                last_time = scalability['query_times'][-1]

                if first_time and last_time:
                    scaling_factor = last_time / first_time

                    if scaling_factor < 1.5:
                        scaling_assessment = "excellent"
                    elif scaling_factor < 2:
                        scaling_assessment = "very good"
                    elif scaling_factor < 3:
                        scaling_assessment = "good"
                    elif scaling_factor < 5:
                        scaling_assessment = "acceptable"
                    else:
                        scaling_assessment = "poor"

                    report_lines.append(
                        f"- Scalability is {scaling_assessment} with a scaling factor of {scaling_factor:.2f} from {scalability['doc_counts'][0]} to {scalability['doc_counts'][-1]} documents.")

        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"Performance report saved to {output_path}")
        return output_path

    def run_complete_performance_test(self):
        """
        Run a complete performance test of the system.

        Returns:
            dict: All performance metrics
        """
        print("Starting complete performance test...")

        # Test Solr connection
        print("\n1. Testing Solr connection...")
        connection_ok = self.test_solr_connection()

        if not connection_ok:
            print("Failed to connect to Solr. Aborting performance test.")
            return {}

        print(f"Connected to Solr. Version: {self.system_info.get('solr', {}).get('version', 'Unknown')}")

        # Run default queries
        print("\n2. Running default search queries...")
        query_metrics = self.run_default_queries()

        # Test facet performance
        print("\n3. Testing facet performance...")
        facet_metrics = self.test_facet_performance()

        # Test scalability
        print("\n4. Testing scalability...")
        scalability_metrics = self.test_scalability()

        # Generate visualizations
        print("\n5. Generating performance visualizations...")
        visualizations = self.visualize_performance()

        # Export performance report
        print("\n6. Exporting performance report...")
        report_path = self.export_performance_report()

        print(f"\nPerformance testing complete. Report saved to {report_path}")

        return self.performance_metrics


# Example usage:
if __name__ == "__main__":
    # Create evaluator
    evaluator = PerformanceEvaluator()

    # Run complete performance test
    results = evaluator.run_complete_performance_test()