# Search System Performance Report
Generated on: 2025-03-31 14:10:33

## System Information
System information not available.

## Query Performance
|   ID | Description               |   Results | Avg Time (ms)   | Min Time (ms)   | Max Time (ms)   |
|-----:|:--------------------------|----------:|:----------------|:----------------|:----------------|
|    1 | All documents             |     60153 | 9.38            | 5.47            | 12.54           |
|    2 | Netflix mentions          |         0 | 6.69            | 5.51            | 8.28            |
|    3 | Streaming price increases |         0 | 13.59           | 11.90           | 14.81           |
|    4 | Technical issues          |         0 | 10.42           | 8.84            | 11.79           |
|    5 | Content quality           |         0 | N/A             | N/A             | N/A             |

## Facet Query Performance
| Facet Type           |   Avg Time (ms) |   Min Time (ms) |   Max Time (ms) |
|:---------------------|----------------:|----------------:|----------------:|
| Platform facets      |            7.77 |            6.59 |            9.19 |
| Sentiment facets     |            9.88 |            9.19 |           10.33 |
| Date range facets    |           23.89 |           22    |           24.93 |
| Feature pivot facets |            5.11 |            4.37 |            5.96 |

## Scalability Performance
|   Document Count |   Query Time (ms) |   Facet Time (ms) |
|-----------------:|------------------:|------------------:|
|            12030 |              8.16 |             11.64 |
|            24061 |              9.81 |              9.55 |
|            36091 |              9.8  |             11.47 |
|            48122 |             10.9  |              9.87 |
|            60153 |              9.6  |             10.65 |

## Performance Visualizations
See the following visualization files:
- Query Performance: `query_performance.png`
- Facet Performance: `facet_performance.png`
- Scalability: `scalability.png`

## Conclusions
- Overall query performance is excellent with an average response time of 10.02 ms.
- Scalability is excellent with a scaling factor of 1.18 from 12030 to 60153 documents.