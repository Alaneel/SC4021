"""
Mock crawler for generating synthetic EV-related news articles
"""
import pandas as pd
import random
import time
from datetime import datetime, timedelta
import os
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Mock data for EV-related content
EV_BRANDS = [
    "Tesla", "Rivian", "Lucid Motors", "Chevrolet", "Ford", "Nissan",
    "Volkswagen", "Hyundai", "Kia", "Audi", "Porsche", "BMW"
]

EV_MODELS = {
    "Tesla": ["Model S", "Model 3", "Model Y", "Model X", "Cybertruck"],
    "Rivian": ["R1T", "R1S"],
    "Lucid Motors": ["Air"],
    "Chevrolet": ["Bolt EV", "Bolt EUV"],
    "Ford": ["Mustang Mach-E", "F-150 Lightning"],
    "Nissan": ["Leaf", "Ariya"],
    "Volkswagen": ["ID.4"],
    "Hyundai": ["Ioniq 5", "Kona Electric"],
    "Kia": ["EV6", "Niro EV"],
    "Audi": ["e-tron", "Q4 e-tron"],
    "Porsche": ["Taycan"],
    "BMW": ["i3", "i4", "iX"]
}

EV_FEATURES = [
    "range", "battery", "charging", "autopilot", "self-driving", "acceleration",
    "price", "cost", "maintenance", "reliability", "comfort", "space", "technology",
    "software", "update", "interior", "exterior", "design", "handling", "noise"
]

SENTIMENT_TEMPLATES = {
    "positive": [
        "I'm really impressed with the {model} from {brand}. The {feature} is outstanding!",
        "Just test drove the {brand} {model} and it exceeded all my expectations. {feature} was amazing.",
        "After owning the {model} for a month, I can confidently say it's the best EV in its class. The {feature} is exceptional.",
        "The new {brand} {model} has incredible {feature}, making it a game changer in the EV market.",
        "{brand}'s commitment to improving {feature} in the {model} shows why they're leading the industry."
    ],
    "negative": [
        "Disappointed with my {brand} {model}. The {feature} is nowhere near what was advertised.",
        "Had to return my {model} because of issues with the {feature}. Not what I expected from {brand}.",
        "The {feature} on the {brand} {model} is frustratingly bad compared to competitors.",
        "Wouldn't recommend the {brand} {model}. The {feature} problems make it not worth the price.",
        "{brand} really dropped the ball with the {feature} on their new {model}. Major letdown."
    ],
    "neutral": [
        "The {feature} on the {brand} {model} is adequate, but nothing special compared to other EVs.",
        "Just read a comparison between EVs and the {brand} {model} seems average in terms of {feature}.",
        "The {model}'s {feature} meets industry standards, but {brand} could do more to innovate here.",
        "Considering a {brand} {model} but still researching how the {feature} compares to other options.",
        "The {feature} in the new {model} from {brand} has both pros and cons worth considering."
    ]
}

NEWS_SOURCES = [
    {"id": "techcrunch", "name": "TechCrunch"},
    {"id": "wired", "name": "Wired"},
    {"id": "theverge", "name": "The Verge"},
    {"id": "cnet", "name": "CNET"},
    {"id": "bbc-news", "name": "BBC News"},
    {"id": "cnn", "name": "CNN"},
    {"id": "bloomberg", "name": "Bloomberg"},
    {"id": "business-insider", "name": "Business Insider"},
    {"id": "engadget", "name": "Engadget"},
    {"id": "reuters", "name": "Reuters"}
]

AUTHORS = [
    "John Smith", "Emma Johnson", "Michael Brown", "Sarah Davis", "Robert Wilson",
    "Jennifer Martinez", "David Garcia", "Lisa Rodriguez", "James Anderson", "Patricia Thomas"
]

# Additional paragraphs to make content more realistic
ADDITIONAL_PARAGRAPHS = [
    "Electric vehicles are rapidly gaining market share as consumers prioritize sustainability and governments introduce stricter emissions regulations.",
    "The charging infrastructure continues to expand, making EV ownership more practical for a wider range of consumers.",
    "Battery technology improvements have addressed many of the range anxiety concerns that previously limited EV adoption.",
    "Many analysts predict that EVs will reach price parity with internal combustion vehicles within the next few years.",
    "The total cost of ownership for EVs is often lower than comparable gas vehicles when considering fuel and maintenance savings.",
    "New entrants to the EV market are challenging established automakers with innovative designs and features.",
    "Software updates are becoming a key differentiator in the EV market, with some brands offering regular over-the-air improvements.",
    "The integration of renewable energy sources with EV charging is creating new opportunities for sustainable transportation.",
    "Range remains a top consideration for potential EV buyers, especially those who frequently travel long distances.",
    "The secondhand EV market is growing, making electric vehicles more accessible to budget-conscious consumers."
]


class MockNewsAPICrawler:
    """
    Mock crawler for generating synthetic EV-related news articles
    """

    def __init__(self):
        """Initialize the mock crawler"""
        self.data = []
        self.EV_BRANDS = EV_BRANDS
        logger.info("Mock News API crawler initialized")


    def _generate_mock_article(self, search_query=None):
        """
        Generate a mock article with realistic EV-related content

        Args:
            search_query (str): Optional search query to include in the article

        Returns:
            dict: Mock article data
        """
        # Select a random brand and model
        brand = random.choice(EV_BRANDS)
        model = random.choice(EV_MODELS[brand])
        feature = random.choice(EV_FEATURES)

        # Determine sentiment and select a template
        sentiment = random.choice(["positive", "negative", "neutral"])
        template = random.choice(SENTIMENT_TEMPLATES[sentiment])

        # Create base content using the template
        base_content = template.format(brand=brand, model=model, feature=feature)

        # Add some additional random content for variety
        additional_paragraphs = random.sample(ADDITIONAL_PARAGRAPHS, k=min(3, len(ADDITIONAL_PARAGRAPHS)))
        full_content = f"{base_content} {' '.join(additional_paragraphs)}"

        # Create a title
        title_templates = {
            "positive": [f"Review: The {brand} {model} Impresses", f"Why the {brand} {model} is Worth Considering",
                         f"The {feature} on the {brand} {model} Sets New Standards"],
            "negative": [f"Issues Found in the {brand} {model}", f"Concerns about the {brand} {model}'s {feature}",
                         f"Why the {brand} {model} Disappointed in Our Tests"],
            "neutral": [f"{brand} {model}: A Balanced Look", f"Comparing the {brand} {model} with Competitors",
                        f"Is the {brand} {model}'s {feature} Worth the Price?"]
        }
        title = random.choice(title_templates[sentiment])

        # Generate a publication date (within the last 30 days)
        days_ago = random.randint(0, 30)
        pub_date = datetime.now() - timedelta(days=days_ago)
        formatted_date = pub_date.strftime('%Y-%m-%d %H:%M:%S')

        # Create a score based on recency
        score = max(1, 100 - (days_ago * 3))

        # Select a random source
        source = random.choice(NEWS_SOURCES)

        # Generate a unique ID
        article_id = f"mock_news_{source['id']}_{int(time.time())}_{random.randint(1000, 9999)}"

        # Generate a realistic URL
        url = f"https://{source['id'].replace('-', '')}.com/articles/{pub_date.year}/{pub_date.month:02d}/{pub_date.day:02d}/ev-{brand.lower().replace(' ', '-')}-{model.lower().replace(' ', '-')}"

        # Create the full article data
        article_data = {
            'id': article_id,
            'type': 'news',
            'title': title,
            'text': full_content,
            'author': random.choice(AUTHORS),
            'created_utc': formatted_date,
            'score': score,
            'subreddit': 'news',  # Use 'news' as a platform indicator
            'url': url,
            'source_name': source['name'],
            'source_id': source['id'],
            'description': f"An analysis of the {brand} {model}, focusing on its {feature} and overall performance in the EV market.",
            'search_query': search_query or '',
            'platform': 'news'
        }

        return article_data

    def generate_mock_data(self, num_articles=100):
        """
        Generate a specified number of mock articles

        Args:
            num_articles (int): Number of articles to generate

        Returns:
            int: Number of articles generated
        """
        initial_count = len(self.data)

        # Generate mock articles
        for _ in tqdm(range(num_articles), desc="Generating mock articles"):
            # Randomly select a search query
            search_query = random.choice([
                "electric vehicle", "EV", random.choice(EV_BRANDS),
                f"{random.choice(EV_BRANDS)} {random.choice(EV_MODELS[random.choice(EV_BRANDS)])}"
            ])

            article = self._generate_mock_article(search_query)
            self.data.append(article)

        new_articles = len(self.data) - initial_count
        logger.info(f"Generated {new_articles} mock articles")

        return new_articles

    def search_articles(self, query, days_back=30, max_results=100, language='en'):
        """
        Mock implementation of search_articles that generates mock data

        Args:
            query (str): Search query
            days_back (int): Number of days to look back
            max_results (int): Maximum number of articles to retrieve
            language (str): Language of articles to retrieve

        Returns:
            int: Number of new items generated
        """
        logger.info(f"Mock searching News API for: '{query}'...")
        return self.generate_mock_data(max_results)

    def search_multiple_queries(self, queries, max_results_per_query=100, days_back=30):
        """
        Mock implementation of search_multiple_queries

        Args:
            queries (list): List of search queries
            max_results_per_query (int): Maximum articles per query
            days_back (int): Number of days to look back

        Returns:
            int: Total number of items generated
        """
        initial_count = len(self.data)

        total_articles = len(queries) * max_results_per_query
        # Limit to a reasonable number for testing
        if total_articles > 1000:
            total_articles = 1000

        self.generate_mock_data(total_articles)

        new_items = len(self.data) - initial_count
        logger.info(f"Total new mock items generated: {new_items}")
        return new_items

    def save_intermediate_results(self):
        """Mock implementation of save_intermediate_results"""
        if not self.data:
            return

        # Create timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            from config.app_config import RAW_DATA_DIR
            filename = os.path.join(RAW_DATA_DIR, f"mock_ev_opinions_intermediate_{timestamp}.csv")

            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            df = pd.DataFrame(self.data)
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(df)} intermediate records to {filename}")
        except Exception as e:
            logger.error(f"Error saving intermediate results: {str(e)}")

    def save_to_csv(self, filename=None):
        """
        Save the generated mock data to a CSV file

        Args:
            filename (str): Output filename

        Returns:
            pandas.DataFrame: The saved dataframe
        """
        if not self.data:
            logger.warning("No data to save")
            return None

        if filename is None:
            try:
                from config.app_config import RAW_DATA_DIR
                filename = os.path.join(RAW_DATA_DIR, f"mock_news_ev_opinions_{datetime.now().strftime('%Y%m%d')}.csv")
            except ImportError:
                filename = f"mock_news_ev_opinions_{datetime.now().strftime('%Y%m%d')}.csv"

        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        df = pd.DataFrame(self.data)

        # Remove duplicates by URL
        df = df.drop_duplicates(subset=['url'])

        # Save to CSV
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(df)} mock records to {filename}")

        # Print some statistics
        total_records = len(df)
        logger.info(f"Total records: {total_records}")

        # Word count stats
        df['word_count'] = df['text'].str.split().str.len()
        total_words = df['word_count'].sum()
        unique_words = len(set(' '.join(df['text'].dropna()).split()))

        logger.info(f"Total words: {total_words}")
        logger.info(f"Unique words: {unique_words}")
        logger.info(f"Average words per record: {total_words / total_records:.1f}")

        return df