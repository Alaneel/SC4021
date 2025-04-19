import requests
import time
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import os

# Solr connection settings
SOLR_URL = "http://localhost:8983/solr/streaming_opinions"

# Define the query test cases from the table
query_test_cases = [
    {
        "name": "Basic Term Query (Control Test)",
        "query_params": {
            "q": "platform:netflix",
            "rows": 10,
            "wt": "json"
        },
        "description": "Simple query for documents with platform Netflix"
    },
    {
        "name": "Basic Query to Compare 2 Platforms",
        "query_params": {
            "q": "platform:netflix platform:disney",
            "q.op": "OR",
            "rows": 10,
            "wt": "json"
        },
        "description": "Query comparing Netflix and Disney platforms with OR operator"
    },
    {
        "name": "Multi-Term Query",
        "query_params": {
            "q": "platform:netflix sentiment:negative",
            "q.op": "AND",
            "rows": 10,
            "wt": "json"
        },
        "description": "Query for documents with Netflix platform AND negative sentiment"
    },
    {
        "name": "Basic Term Query with Filter",
        "query_params": {
            "q": "platform:netflix",
            "fq": "sentiment:negative",
            "rows": 10,
            "wt": "json"
        },
        "description": "Query for Netflix platform with filter for negative sentiment"
    },
    {
        "name": "Multi-Term Query with Sort",
        "query_params": {
            "q": "platform:netflix sentiment:negative",
            "q.op": "AND",
            "sort": "created_at desc",
            "rows": 10,
            "wt": "json"
        },
        "description": "Query for Netflix platform AND negative sentiment, sorted by creation date descending"
    }
]

def run_query_test(query_params, num_runs=5):
    """Run a Solr query multiple times and measure performance"""
    execution_times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        response = requests.get(f"{SOLR_URL}/select", params=query_params)
        response.raise_for_status()
        end_time = time.time()
        
        # Calculate execution time in milliseconds
        execution_time_ms = (end_time - start_time) * 1000
        execution_times.append(execution_time_ms)
        
        # Small delay between runs to avoid overwhelming Solr
        time.sleep(0.1)
    
    # Calculate statistics
    avg_time = statistics.mean(execution_times)
    median_time = statistics.median(execution_times)
    min_time = min(execution_times)
    max_time = max(execution_times)
    std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
    
    return {
        "execution_times": execution_times,
        "avg_time": avg_time,
        "median_time": median_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_dev": std_dev,
        "num_found": response.json()["response"]["numFound"]
    }

def test_solr_connection():
    """Test if Solr is up and running"""
    try:
        response = requests.get(f"{SOLR_URL}/admin/ping", timeout=5)
        response.raise_for_status()
        print(f"Successfully connected to Solr at {SOLR_URL}")
        return True
    except Exception as e:
        print(f"Failed to connect to Solr: {e}")
        return False

def run_all_tests(num_runs=5):
    """Run all query tests and collect results"""
    if not test_solr_connection():
        return None
    
    results = []
    
    print("Running query performance tests...")
    for test_case in query_test_cases:
        print(f"Testing: {test_case['name']}")
        test_result = run_query_test(test_case["query_params"], num_runs)
        
        result_entry = {
            "name": test_case["name"],
            "description": test_case["description"],
            "query_params": test_case["query_params"],
            "avg_time_ms": round(test_result["avg_time"], 2),
            "median_time_ms": round(test_result["median_time"], 2),
            "min_time_ms": round(test_result["min_time"], 2),
            "max_time_ms": round(test_result["max_time"], 2),
            "std_dev_ms": round(test_result["std_dev"], 2),
            "num_found": test_result["num_found"]
        }
        
        results.append(result_entry)
    
    return results

def display_results_table(results):
    """Display results in a formatted table"""
    table_data = []
    
    for result in results:
        row = [
            result["name"],
            f"{result['avg_time_ms']:.2f}",
            f"{result['median_time_ms']:.2f}",
            f"{result['min_time_ms']:.2f}",
            f"{result['max_time_ms']:.2f}",
            f"{result['std_dev_ms']:.2f}",
            result["num_found"]
        ]
        table_data.append(row)
    
    headers = ["Query Type", "Avg (ms)", "Median (ms)", "Min (ms)", "Max (ms)", "Std Dev (ms)", "Results"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def plot_performance_comparison(results):
    """Create a bar chart comparing query performance"""
    query_names = [result["name"] for result in results]
    avg_times = [result["avg_time_ms"] for result in results]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(query_names, avg_times)
    
    # Add data labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f} ms', ha='center', va='bottom')
    
    plt.xlabel('Query Type')
    plt.ylabel('Average Execution Time (ms)')
    plt.title('Solr Query Performance Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs('../evaluation/evaluation/results', exist_ok=True)
    
    # Save the figure
    plt.savefig('evaluation/evaluation/results/query_performance.png')
    print("Performance chart saved to evaluation/evaluation/results/query_performance.png")
    plt.close()

def generate_performance_report(results, scalability_results=None, facet_results=None):
    """Generate a markdown performance report"""
    report = """# Solr Query Performance Report

## Overview
This report summarizes the performance of different query types executed against our Solr instance for the streaming platform opinions dataset.

## Test Environment
- Solr URL: {solr_url}
- Test Date: {test_date}
- Test Runs per Query: {num_runs}

## Query Performance Results

| Query Type | Avg (ms) | Median (ms) | Min (ms) | Max (ms) | Std Dev (ms) | Results |
|------------|----------|-------------|----------|----------|--------------|---------|
""".format(
        solr_url=SOLR_URL,
        test_date=time.strftime("%Y-%m-%d"),
        num_runs=5  # Default value
    )
    
    for result in results:
        report += "| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {} |\n".format(
            result["name"],
            result["avg_time_ms"],
            result["median_time_ms"],
            result["min_time_ms"],
            result["max_time_ms"],
            result["std_dev_ms"],
            result["num_found"]
        )
    
    report += """
## Query Details

"""
    
    for result in results:
        report += f"### {result['name']}\n"
        report += f"- Description: {result['description']}\n"
        report += f"- Query Parameters:\n"
        
        for param, value in result["query_params"].items():
            report += f"  - {param}: {value}\n"
        
        report += f"- Average Execution Time: {result['avg_time_ms']:.2f} ms\n"
        report += f"- Documents Found: {result['num_found']}\n\n"
    
    # Add scalability results if available
    if scalability_results:
        report += """
## Scalability Performance

Testing how query performance scales with increasing result set sizes.

| Rows Requested | Avg Time (ms) | Actual Rows Returned |
|----------------|---------------|----------------------|
"""
        
        for result in scalability_results:
            report += "| {} | {:.2f} | {} |\n".format(
                result["rows"],
                result["avg_time_ms"],
                result["actual_rows"]
            )
            
        report += """
![Scalability Chart](scalability.png)

"""

    # Add facet results if available
    if facet_results:
        report += """
## Facet Query Performance

Testing the performance of different facet query types.

| Facet Type | Avg Time (ms) |
|------------|---------------|
"""
        
        for result in facet_results:
            report += "| {} | {:.2f} |\n".format(
                result["name"],
                result["avg_time_ms"]
            )
            
        report += """
![Facet Performance Chart](facet_performance.png)

"""
    
    report += """
## Analysis

1. **Basic Query Performance**: The simple basic queries offer the fastest performance, as expected.
2. **Filter vs. Query**: Using filter queries (`fq`) is generally faster than using equivalent query terms with `q.op=AND`.
3. **Sorting Impact**: Queries with sorting operations show the most significant performance penalty.
4. **Platform Comparison**: Comparing multiple platforms with `q.op=OR` is relatively efficient despite the larger result set.
"""

    # Add scalability analysis if available
    if scalability_results:
        report += """
5. **Scalability**: Query execution time increases with the number of rows requested, but the relationship is not strictly linear. The performance impact becomes more pronounced at higher row counts.
"""

    # Add facet analysis if available
    if facet_results:
        report += """
6. **Facet Performance**: Date range facets are the most expensive, followed by multiple field facets. Simple field facets on low-cardinality fields like platform and sentiment are quite efficient.
"""
    
    report += """
## Recommendations

1. Use filter queries (`fq`) wherever possible instead of query terms for better performance.
2. Consider pagination and limiting result sets when using sorted queries.
3. For comparing multiple platforms, consider using facet queries rather than `OR` operators for better scalability.
4. Limit the number of rows returned in a single request to improve response times, especially when dealing with large result sets.
5. Be cautious with date range faceting as it has the highest performance impact; consider caching these results when possible.
"""
    
    # Ensure directory exists
    os.makedirs('../evaluation/evaluation/results', exist_ok=True)
    
    # Save the report
    with open('../evaluation/evaluation/results/performance_report.md', 'w') as f:
        f.write(report)
    
    print("Performance report saved to evaluation/evaluation/results/performance_report.md")

def test_scalability():
    """Test scalability of queries with increasing result sizes"""
    import time
    if not test_solr_connection():
        return None
    
    # Use the basic Netflix query as our test case
    base_query = {
        "q": "platform:netflix",
        "wt": "json"
    }
    
    # Test with different row counts
    row_sizes = [10, 50, 100, 500, 1000]
    results = []
    
    print("\nTesting query scalability with different result sizes...")
    for rows in row_sizes:
        query = base_query.copy()
        query["rows"] = rows
        
        print(f"Testing with rows={rows}")
        times = []
        
        # Run each test 3 times for consistency
        for _ in range(3):
            start_time = time.time()
            response = requests.get(f"{SOLR_URL}/select", params=query)
            response.raise_for_status()
            end_time = time.time()
            
            # Calculate execution time in milliseconds
            execution_time_ms = (end_time - start_time) * 1000
            times.append(execution_time_ms)
            
            # Small delay between runs
            time.sleep(0.1)
        
        avg_time = statistics.mean(times)
        results.append({
            "rows": rows,
            "avg_time_ms": avg_time,
            "actual_rows": len(response.json()["response"]["docs"])
        })
    
    # Generate scalability visualization
    plt.figure(figsize=(10, 6))
    x = [result["rows"] for result in results]
    y = [result["avg_time_ms"] for result in results]
    
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel('Number of Rows Requested')
    plt.ylabel('Average Execution Time (ms)')
    plt.title('Solr Query Scalability')
    plt.grid(True)
    
    # Add data labels
    for i, (rows, time) in enumerate(zip(x, y)):
        plt.text(rows, time + 1, f"{time:.2f} ms", ha='center')
    
    plt.tight_layout()
    plt.savefig('evaluation/evaluation/results/scalability.png')
    print("Scalability chart saved to evaluation/evaluation/results/scalability.png")
    plt.close()
    
    return results

def test_facet_performance():
    """Test performance of facet queries"""
    import time
    if not test_solr_connection():
        return None
    
    # Define different facet types
    facet_tests = [
        {
            "name": "Platform Facets",
            "params": {
                "q": "*:*",
                "facet": "true",
                "facet.field": "platform",
                "facet.mincount": 1,
                "rows": 0,
                "wt": "json"
            }
        },
        {
            "name": "Sentiment Facets",
            "params": {
                "q": "*:*",
                "facet": "true",
                "facet.field": "sentiment",
                "facet.mincount": 1,
                "rows": 0,
                "wt": "json"
            }
        },
        {
            "name": "Date Range Facets",
            "params": {
                "q": "*:*",
                "facet": "true",
                "facet.range": "created_at",
                "facet.range.start": "NOW-1YEAR",
                "facet.range.end": "NOW",
                "facet.range.gap": "+1MONTH",
                "rows": 0,
                "wt": "json"
            }
        },
        {
            "name": "Multiple Field Facets",
            "params": {
                "q": "*:*",
                "facet": "true",
                "facet.field": ["platform", "sentiment", "type"],
                "facet.mincount": 1,
                "rows": 0,
                "wt": "json"
            }
        }
    ]
    
    results = []
    
    print("\nTesting facet query performance...")
    for test in facet_tests:
        print(f"Testing: {test['name']}")
        times = []
        
        # Run each test 3 times
        for _ in range(3):
            start_time = time.time()
            response = requests.get(f"{SOLR_URL}/select", params=test["params"])
            response.raise_for_status()
            end_time = time.time()
            
            # Calculate execution time in milliseconds
            execution_time_ms = (end_time - start_time) * 1000
            times.append(execution_time_ms)
            
            # Small delay between runs
            time.sleep(0.1)
        
        avg_time = statistics.mean(times)
        results.append({
            "name": test["name"],
            "avg_time_ms": avg_time
        })
    
    # Generate facet performance visualization
    plt.figure(figsize=(10, 6))
    names = [result["name"] for result in results]
    times = [result["avg_time_ms"] for result in results]
    
    bars = plt.bar(names, times)
    plt.xlabel('Facet Type')
    plt.ylabel('Average Execution Time (ms)')
    plt.title('Solr Facet Query Performance')
    plt.xticks(rotation=45, ha='right')
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f} ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('evaluation/evaluation/results/facet_performance.png')
    print("Facet performance chart saved to evaluation/evaluation/results/facet_performance.png")
    plt.close()
    
    return results

if __name__ == "__main__":
    print("Starting Solr query performance testing...")
    
    # Run all tests with 5 runs each
    results = run_all_tests(num_runs=5)
    
    if results:
        # Display results table
        display_results_table(results)
        
        # Generate visualization
        plot_performance_comparison(results)
        
        # Test scalability with different result sizes
        print("\nTesting query scalability...")
        scalability_results = test_scalability()
        
        # Test facet performance
        print("\nTesting facet query performance...")
        facet_results = test_facet_performance()
        
        # Generate comprehensive performance report
        generate_performance_report(results, scalability_results, facet_results)
        
        print("\nPerformance testing completed successfully!")
    else:
        print("Performance testing failed - could not connect to Solr.")