"""Configuration settings for social media crawlers."""

# Crawler default settings
DEFAULT_REDDIT_LIMIT = 100

# Subreddits to crawl
SUBREDDITS = [
    'Netflix', 'DisneyPlus', 'Hulu', 'PrimeVideo', 'cordcutters', 'StreamingBestOf',
    'HBOMax', 'appletv', 'peacock', 'paramountplus'
]

# Output file paths
REDDIT_OUTPUT_CSV = "../data/reddit_streaming_data.csv"
COMBINED_OUTPUT_CSV = "../data/streaming_opinions_dataset.csv"

# Streaming platforms for detection
STREAMING_PLATFORMS = {
    'netflix': ['netflix', 'netflix\'s'],
    'disney+': ['disney+', 'disney plus', 'disneyplus'],
    'hbo max': ['hbo max', 'hbomax', 'hbo'],
    'amazon prime': ['amazon prime', 'prime video', 'primevideo'],
    'hulu': ['hulu', 'hulu\'s'],
    'apple tv+': ['apple tv+', 'apple tv plus', 'appletv+'],
    'peacock': ['peacock'],
    'paramount+': ['paramount+', 'paramount plus']
}