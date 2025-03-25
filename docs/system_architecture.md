# Streaming Opinion Search Engine - System Architecture

## Overview

The Streaming Opinion Search Engine is a comprehensive system for collecting, analyzing, and searching opinions about various streaming services. The architecture follows a modular design with clearly separated components for crawling, processing, classification, indexing, search, and visualization.

## System Components

### 1. Data Collection Subsystem

The data collection subsystem is responsible for gathering opinions from various sources:

#### Reddit Crawler
- **Purpose**: Collects posts and comments from streaming-related subreddits
- **Components**:
  - `RedditCrawler` class: Main crawler implementation using PRAW
  - Subreddit configuration manager
  - Rate limiting and error handling
- **Data Flow**:
  - Connects to Reddit API
  - Iterates through configured subreddits
  - Extracts posts and comments
  - Stores raw data in CSV format

#### Twitter Crawler
- **Purpose**: Gathers tweets containing streaming service keywords
- **Components**:
  - `TwitterCrawler` class: Implementation using Tweepy
  - Query configuration manager
  - Rate limiting and pagination handler
- **Data Flow**:
  - Connects to Twitter API
  - Executes configured search queries
  - Filters out retweets and non-English content
  - Stores raw data in CSV format

#### Data Integration
- **Purpose**: Combines data from multiple sources into a unified format
- **Components**:
  - `combine_datasets` function: Harmonizes schema and merges data
  - ID generation and deduplication logic
- **Data Flow**:
  - Reads raw source-specific data
  - Transforms to common schema
  - Assigns unique IDs
  - Creates unified dataset

### 2. Data Processing Subsystem

The data processing subsystem cleans and enriches the raw data:

#### Text Processing Pipeline
- **Purpose**: Normalizes and cleans text data
- **Components**:
  - `EnhancedDataProcessor` class: Central processing controller
  - Text cleaning functions: URL removal, normalization, etc.
  - Language detection: Filters non-English content
- **Data Flow**:
  - Takes raw text input
  - Applies cleaning steps
  - Normalizes structures
  - Produces cleaned text for analysis

#### Feature Extraction
- **Purpose**: Extracts domain-specific features from text
- **Components**:
  - Platform detection: Identifies mentioned streaming services
  - Feature extractors: Content quality, pricing, UI/UX, etc.
  - Metadata enrichment: Creation date normalization, etc.
- **Data Flow**:
  - Analyzes cleaned text
  - Identifies platforms and features
  - Calculates feature scores
  - Adds metadata to records

#### Duplicate Detection
- **Purpose**: Identifies and removes duplicate or near-duplicate content
- **Components**:
  - TF-IDF vectorizer: Converts text to vector representation
  - Cosine similarity calculator: Measures document similarity
  - Threshold-based filtering: Removes similar documents
- **Data Flow**:
  - Vectorizes document content
  - Calculates pairwise similarity
  - Flags or removes duplicates

### 3. Classification Subsystem

The classification subsystem analyzes opinions to determine sentiment and categorize content:

#### Sentiment Analysis
- **Purpose**: Classifies sentiment as positive, negative, or neutral
- **Components**:
  - Lexicon-based analyzer: VADER sentiment analyzer
  - Machine learning classifier: Random Forest model
  - Hybrid integration system: Combines both approaches
- **Data Flow**:
  - Takes cleaned text as input
  - Calculates sentiment scores
  - Determines sentiment category
  - Adds sentiment data to records

#### Feature-Specific Analysis
- **Purpose**: Analyzes opinions about specific streaming service features
- **Components**:
  - Content quality analyzer
  - Pricing analyzer
  - UI/UX analyzer
  - Technical performance analyzer
  - Customer service analyzer
- **Data Flow**:
  - Extracts feature-specific text segments
  - Analyzes sentiment for each feature
  - Assigns feature scores
  - Adds feature data to records

#### Enhanced Classification
- **Purpose**: Improves classification accuracy through advanced techniques
- **Components**:
  - Word Sense Disambiguation module
  - Named Entity Recognition system
  - Aspect-Based Sentiment Analysis
  - Sarcasm detection
- **Data Flow**:
  - Applies enhancement to text analysis
  - Refines sentiment and feature scores
  - Updates classification results

### 4. Indexing Subsystem

The indexing subsystem makes the data searchable:

#### Solr Integration
- **Purpose**: Indexes processed data in Apache Solr
- **Components**:
  - Schema configuration: Field definitions and types
  - Indexing script: Loads data into Solr
  - Solr configuration: Core settings and optimization
- **Data Flow**:
  - Reads processed data
  - Maps to Solr schema
  - Indexes in Solr
  - Commits and optimizes

#### Search Enhancement
- **Purpose**: Improves search capabilities beyond basic functionality
- **Components**:
  - Custom query parsers: Handle domain-specific syntax
  - Scoring functions: Boost relevant results
  - Faceting configuration: Enable multi-dimensional filtering
- **Data Flow**:
  - Configures Solr extensions
  - Customizes query and response handling
  - Optimizes for streaming opinion search

### 5. Web Application

The web application provides the user interface:

#### Flask Backend
- **Purpose**: Serves the web application and handles requests
- **Components**:
  - Flask application: Main web server
  - Route handlers: Process web requests
  - Search controllers: Interface with Solr
  - Visualization generators: Create data visualizations
- **Data Flow**:
  - Receives user requests
  - Processes search queries
  - Fetches results from Solr
  - Generates response with results and visualizations

#### Frontend Interface
- **Purpose**: Provides user interface for search and exploration
- **Components**:
  - HTML templates: Page structure
  - CSS styles: Visual design
  - JavaScript: Interactive elements
  - Plotly charts: Data visualizations
- **Data Flow**:
  - Renders search interface
  - Accepts user input
  - Displays search results
  - Presents visualizations

#### Advanced Features
- **Purpose**: Provides enhanced search and exploration capabilities
- **Components**:
  - Interactive feedback system
  - Timeline search and visualization
  - Word cloud generator
  - Multi-faceted search interface
- **Data Flow**:
  - Processes advanced user interactions
  - Generates specialized visualizations
  - Provides insights beyond basic search

### 6. Evaluation Framework

The evaluation framework assesses system performance:

#### Sentiment Classifier Evaluation
- **Purpose**: Measures classification accuracy and performance
- **Components**:
  - Evaluation dataset creation tools
  - Metrics calculation: Precision, recall, F1
  - Ablation study framework
- **Data Flow**:
  - Tests classifier on evaluation data
  - Calculates performance metrics
  - Generates evaluation reports

#### Search Performance Testing
- **Purpose**: Measures search speed and result quality
- **Components**:
  - Query performance tester
  - Relevance assessment tools
  - Scalability testing framework
- **Data Flow**:
  - Executes test queries
  - Measures response times
  - Assesses result relevance
  - Tests system at different scales

## Component Interactions

### Data Flow Pipeline

1. **Collection → Processing**:
   - Crawlers collect raw data and store in CSV format
   - Processing pipeline reads raw data
   - Text cleaning and normalization applied
   - Feature extraction and enrichment performed

2. **Processing → Classification**:
   - Cleaned text passed to classification subsystem
   - Sentiment analysis performed
   - Feature-specific analysis completed
   - Enhanced classification techniques applied

3. **Classification → Indexing**:
   - Processed and classified data prepared for indexing
   - Data mapped to Solr schema
   - Records indexed in Solr
   - Index optimized for search

4. **Indexing → Web Application**:
   - Web application connects to Solr
   - User searches translated to Solr queries
   - Results fetched from Solr
   - Results rendered in web interface

### Key Integration Points

1. **Crawler Configuration Interface**:
   - Allows configuration of data sources
   - Controls crawling parameters
   - Manages API credentials

2. **Data Processing Pipeline**:
   - Provides unified interface for all processing steps
   - Ensures consistent data format
   - Manages processing workflow

3. **Classification Service API**:
   - Exposes classification capabilities to other components
   - Provides batch and real-time classification
   - Supports diverse classification tasks

4. **Search API**:
   - Abstracts Solr interaction
   - Provides structured query interface
   - Supports advanced search features

5. **Visualization Framework**:
   - Generates visualizations from search results
   - Supports multiple visualization types
   - Integrates with web interface

## Deployment Architecture

### Development Environment
- Local Solr instance
- Flask development server
- Python virtual environment
- Local file storage

### Production Environment
- **Web Tier**:
  - Flask application behind WSGI server (Gunicorn)
  - Nginx for static content and SSL termination
  - Load balancer for high availability

- **Search Tier**:
  - Solr cluster with multiple shards
  - ZooKeeper for configuration management
  - Separate index nodes for scalability

- **Storage Tier**:
  - File storage for raw and processed data
  - Database for application state and user data
  - Cache layer for frequent queries

- **Processing Tier**:
  - Background workers for data processing
  - Task queue for asynchronous operations
  - Scheduled jobs for data updates

## Performance Considerations

### Scalability
- Horizontal scaling of Solr nodes
- Distributed processing of large datasets
- Partitioning of data by time period
- Caching of frequent queries

### Response Time
- Optimized Solr configuration
- Efficient query construction
- Background processing of intensive tasks
- Frontend optimizations for fast rendering

### Resource Usage
- Batch processing for efficiency
- Incremental updates to reduce processing load
- Resource monitoring and auto-scaling
- Query optimization to reduce computational cost

## Security Considerations

### API Authentication
- Secure storage of API credentials
- Rate limiting to prevent abuse
- Access controls for sensitive endpoints

### Data Privacy
- Anonymization of user identifiers
- Secure storage of raw and processed data
- Limited access to sensitive information

### Application Security
- Input validation and sanitization
- Protection against common web vulnerabilities
- Secure communication with HTTPS

## Extensibility

The system architecture is designed for extensibility:

1. **New Data Sources**:
   - Implement crawler interface for additional platforms
   - Add source-specific processing to data integration

2. **Enhanced Classification**:
   - Add new classification techniques as modules
   - Integrate with existing classification pipeline

3. **Advanced Search Features**:
   - Extend search API with new capabilities
   - Add custom components to Solr configuration

4. **Additional Visualizations**:
   - Implement new visualization types
   - Add to visualization framework

5. **Language Support**:
   - Add language detection and processing
   - Implement language-specific analyzers
   - Support multilingual search

## Architectural Decisions

### Apache Solr vs. Elasticsearch
We chose Apache Solr for our search engine due to its:
- Mature faceting capabilities essential for our multi-dimensional filtering
- Excellent support for text search and relevance tuning
- Strong performance with static document collections
- Lower resource requirements for our deployment environment

### Flask vs. Django
We selected Flask as our web framework because:
- Lightweight nature suited our focused application requirements
- Flexibility for integrating with Solr and visualization libraries
- Simplicity aligned with our minimalist UI approach
- Ease of deployment in various environments

### Hybrid Classification vs. Pure ML
We implemented a hybrid classification approach combining lexicon-based and ML methods because:
- Provides good accuracy without excessive computational requirements
- Offers interpretability through lexicon component
- Adapts to domain-specific language via ML component
- Achieves balance between performance and resource usage

### Batch vs. Real-time Processing
We chose batch processing for initial data with real-time capabilities for user interactions because:
- Efficient processing of large historical datasets
- Responsive experience for interactive search
- Balances computational resources and user experience
- Supports both analytical and interactive use cases

## Conclusion

The Streaming Opinion Search Engine architecture provides a robust, scalable, and extensible framework for collecting, analyzing, and searching opinions about streaming services. The modular design enables independent development and optimization of components while maintaining clear interfaces for integration. The system balances performance, resource usage, and user experience to deliver a responsive and informative search platform.