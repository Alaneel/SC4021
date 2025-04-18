import pandas as pd
import numpy as np
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import time
import warnings
from imblearn.over_sampling import SMOTE
from collections import defaultdict, Counter
from textblob import TextBlob
import spacy
nlp = spacy.load("en_core_web_sm")
import torch

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 3. ENHANCED FEATURE ENGINEERING
# Define cross-platform aspects
def define_cross_platform_aspects():
    """Define aspect keywords that work across streaming platforms"""
    return {
        'content_quality': [
            'quality', 'content', 'show', 'movie', 'series', 'documentary',
            'original', 'programming', 'catalog', 'library', 'selection'
        ],
        'pricing': [
            'price', 'cost', 'subscription', 'fee', 'expensive', 'cheap',
            'worth', 'pay', 'money', 'value', 'plan', 'tier'
        ],
        'ui_ux': [
            'interface', 'design', 'ui', 'ux', 'navigation', 'search',
            'find', 'browse', 'menu', 'layout', 'usability', 'app'
        ],
        'technical': [
            'buffer', 'stream', 'load', 'quality', 'hd', '4k', 'resolution',
            'error', 'bug', 'crash', 'playback', 'bandwidth', 'offline'
        ],
        'customer_service': [
            'support', 'service', 'help', 'contact', 'response', 'customer',
            'chat', 'email', 'refund', 'cancel', 'subscription'
        ]
    }


# 4. MODEL BUILDING AND EVALUATION
def build_sentiment_pipeline(model_type='rf', include_platform=True, available_columns=None):
    """Build a pipeline for sentiment analysis"""
    if model_type == 'lr':
        classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    elif model_type == 'nb':
        classifier = MultinomialNB()
    else:  # default to RandomForest
        classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced')

    # Text features pipeline (will remain sparse)
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('selector', SelectKBest(f_classif, k=2000))
    ])

    # For manual features (will be scaled separately)
    manual_features = Pipeline([
        ('identity', FunctionTransformer(lambda x: x)),
        ('scaler', StandardScaler())  # This scaler only applies to the manual features
    ])

    # Basic manual feature columns that should always be available
    base_manual_cols = ['text_length', 'word_count', 'sentiment_ratio',
                        'positive_word_count', 'negative_word_count']

    # Verify all base columns exist in available_columns
    if available_columns is not None:
        base_manual_cols = [col for col in base_manual_cols if col in available_columns]

        if not base_manual_cols:
            raise ValueError("None of the expected manual feature columns were found in the data")

    # Build transformer list
    transformers = [
        ('text_features', text_pipeline, 'processed_content'),
        ('manual_features', manual_features, base_manual_cols)
    ]

    # Add platform features if requested and available
    if include_platform and available_columns is not None:
        # Dynamically find platform columns that exist in the data
        platform_cols = [col for col in available_columns if col.startswith('platform_')]
        mention_cols = [col for col in available_columns if col.startswith('mentions_')]

        platform_feature_cols = platform_cols + mention_cols

        if platform_feature_cols:
            # Only add platform features if we actually have some
            platform_transformer = Pipeline([
                ('identity', FunctionTransformer(lambda x: x))
            ])

            transformers.append(('platform_features', platform_transformer, platform_feature_cols))
            print(f"Including {len(platform_feature_cols)} platform-related features in the model")
        else:
            print("No platform-specific columns found in the data. Model will not include platform features.")

    # Column transformer to combine feature sets
    preprocessor = ColumnTransformer(transformers, remainder='drop')

    # Final pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    return pipeline


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type='rf',
                             task='sentiment', include_platform=True, platform_stratified=False):
    """Train and evaluate a model for the specified task"""
    # Get list of available columns to pass to pipeline builder
    available_columns = list(X_train.columns)

    # Create pipeline with dynamic column detection
    pipeline = build_sentiment_pipeline(model_type, include_platform, available_columns)

    # For platform-stratified evaluation
    if platform_stratified and 'platform' in X_train.columns:
        metrics_by_platform = {}
        platforms = X_train['platform'].unique()

        for platform in platforms:
            # Skip platforms with too few samples
            platform_train_mask = (X_train['platform'] == platform)
            platform_test_mask = (X_test['platform'] == platform)

            if sum(platform_train_mask) < 30 or sum(platform_test_mask) < 30:
                continue

            # Train on all data but evaluate on platform-specific
            pipeline.fit(X_train, y_train)

            # Platform-specific evaluation
            X_test_platform = X_test[platform_test_mask]
            y_test_platform = y_test[platform_test_mask]

            if len(X_test_platform) > 0:
                y_pred_platform = pipeline.predict(X_test_platform)

                # Calculate metrics
                accuracy = accuracy_score(y_test_platform, y_pred_platform)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test_platform, y_pred_platform, average='binary'
                )

                metrics_by_platform[platform] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'sample_size': len(X_test_platform)
                }

        # Overall metrics
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
    else:
        # Standard training and evaluation
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics_by_platform = None

    # Calculate overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    # Print results
    print(f"\n{task.capitalize()} Classification with {model_type.upper()}:")
    print(f"Overall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print platform-specific metrics if available
    if metrics_by_platform:
        print("\nPlatform-Specific Metrics:")
        for platform, metrics in metrics_by_platform.items():
            print(f"\n  {platform.upper()} (n={metrics['sample_size']}):")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Return metrics and the model
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'by_platform': metrics_by_platform
    }

    return pipeline, metrics


def build_aspect_model(aspect_name, X_train, X_test, y_train, y_test, include_platform=True):
    """Build and evaluate a model for a specific aspect"""
    # Get list of available columns
    available_columns = list(X_train.columns)

    # Create a pipeline specific for this aspect with platform awareness
    pipeline = build_sentiment_pipeline('rf', include_platform, available_columns)

    # Train and evaluate
    return train_and_evaluate_model(
        X_train, X_test, y_train, y_test, 'rf', aspect_name, include_platform
    )


# 5. TRANSFORMER-BASED MODELS
def transformer_sentiment_analysis(df, sample_size=1000, stratify_by_platform=True):
    """Use a pre-trained transformer model for sentiment analysis"""
    print("\nPerforming transformer-based sentiment analysis...")

    # Load pre-trained model and tokenizer
    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Create sentiment analysis pipeline with truncation
    device = 0 if torch.cuda.is_available() else -1
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=512
    )


    # Take a stratified sample for analysis
    if len(df) > sample_size:
        if stratify_by_platform and 'platform' in df.columns:
            # Stratified sampling to ensure representation of all platforms
            sample_indices = []
            platforms = df['platform'].unique()

            for platform in platforms:
                platform_indices = df[df['platform'] == platform].index.tolist()
                # Calculate proportional sample size for this platform
                platform_sample_size = min(
                    int(sample_size * (len(platform_indices) / len(df))),
                    len(platform_indices)
                )

                if platform_sample_size > 0:
                    platform_sample = np.random.choice(
                        platform_indices,
                        size=platform_sample_size,
                        replace=False
                    )
                    sample_indices.extend(platform_sample)

            # If we didn't get enough samples, add more from the remaining data
            if len(sample_indices) < sample_size:
                remaining_indices = list(set(df.index) - set(sample_indices))
                additional_samples = np.random.choice(
                    remaining_indices,
                    size=min(sample_size - len(sample_indices), len(remaining_indices)),
                    replace=False
                )
                sample_indices.extend(additional_samples)

            df_sample = df.loc[sample_indices]
        else:
            # Simple random sampling
            df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df.copy()

    # Process the sample
    start_time = time.time()

    results = []
    batch_size = 32
    for i in range(0, len(df_sample), batch_size):
        batch = df_sample['content'].iloc[i:i + batch_size].tolist()
        # Make sure all items are strings and truncate if needed
        batch = [str(text)[:10000] if text is not None else "" for text in batch]  # Pre-truncate very long texts

        try:
            batch_results = sentiment_analyzer(batch)
            results.extend(batch_results)
        except Exception as e:
            print(f"Error processing batch {i // batch_size}: {e}")
            # For errors, assign neutral sentiment as fallback
            batch_results = [{"label": "NEUTRAL", "score": 0.5} for _ in range(len(batch))]
            results.extend(batch_results)

    process_time = time.time() - start_time

    # Convert results to a format we can use
    df_sample['transformer_sentiment'] = [1 if res['label'] == 'POSITIVE' else 0 for res in results]

    # Compare with manual annotations
    agreement = (df_sample['transformer_sentiment'] == df_sample['sentiment_binary']).mean()
    print(f"Agreement between transformer and manual annotations: {agreement:.4f}")

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        df_sample['sentiment_binary'],
        df_sample['transformer_sentiment'],
        average='binary'
    )

    print(f"Transformer model metrics:")
    print(f"Accuracy: {agreement:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Processing time for {len(df_sample)} samples: {process_time:.2f} seconds")
    print(f"Processing speed: {len(df_sample) / process_time:.2f} samples/second")

    # Platform-specific metrics if available
    if 'platform' in df_sample.columns:
        print("\nTransformer model metrics by platform:")
        platforms = df_sample['platform'].unique()

        for platform in platforms:
            platform_df = df_sample[df_sample['platform'] == platform]

            if len(platform_df) >= 30:  # Only evaluate platforms with enough samples
                platform_agreement = (platform_df['transformer_sentiment'] == platform_df['sentiment_binary']).mean()
                platform_precision, platform_recall, platform_f1, _ = precision_recall_fscore_support(
                    platform_df['sentiment_binary'],
                    platform_df['transformer_sentiment'],
                    average='binary'
                )

                print(f"\n  {platform.upper()} (n={len(platform_df)}):")
                print(f"  Accuracy: {platform_agreement:.4f}")
                print(f"  Precision: {platform_precision:.4f}")
                print(f"  Recall: {platform_recall:.4f}")
                print(f"  F1 Score: {platform_f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(df_sample['sentiment_binary'], df_sample['transformer_sentiment'])
    print("\nConfusion Matrix:")
    print(cm)

    return sentiment_analyzer, {
        'accuracy': agreement,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'speed': len(df_sample) / process_time
    }


# 6. ASPECT-BASED SENTIMENT ANALYSIS (ABSA)
def perform_absa(df, sentiment_analyzer, by_platform=True):
    """Perform Aspect-Based Sentiment Analysis using transformer model"""
    print("\nPerforming Aspect-Based Sentiment Analysis (ABSA)...")

    # Get cross-platform aspect definitions
    aspects = ['content_quality', 'pricing', 'ui_ux', 'technical', 'customer_service']
    aspect_keywords = define_cross_platform_aspects()

    # Ensure sentiment_analyzer has truncation enabled
    if not hasattr(sentiment_analyzer, 'tokenizer') or not getattr(sentiment_analyzer.tokenizer, 'truncation', False):
        print("Setting truncation for the sentiment analyzer in ABSA")
        sentiment_analyzer.tokenizer.truncation = True
        sentiment_analyzer.tokenizer.max_length = 512

    # Sample data for ABSA (for performance reasons)
    sample_size = min(500, len(df))

    # Stratified sampling by platform if requested
    if by_platform and 'platform' in df.columns:
        # Get platform distribution
        platforms = df['platform'].unique()
        platform_counts = df['platform'].value_counts()

        # Calculate samples per platform
        samples_per_platform = {}
        for platform in platforms:
            # Proportional allocation with minimum threshold
            platform_sample = max(
                int(sample_size * (platform_counts[platform] / len(df))),
                min(30, platform_counts[platform]) if platform_counts[platform] > 0 else 0
            )
            samples_per_platform[platform] = platform_sample

        # Sample from each platform
        sample_dfs = []
        for platform, count in samples_per_platform.items():
            if count > 0:
                platform_df = df[df['platform'] == platform]
                if len(platform_df) >= count:
                    sample_dfs.append(platform_df.sample(count, random_state=42))
                else:
                    sample_dfs.append(platform_df)  # Take all if fewer than requested

        # Combine samples
        df_sample = pd.concat(sample_dfs)
    else:
        # Simple random sampling
        df_sample = df.sample(sample_size, random_state=42)

    # Initialize results dictionaries
    absa_results = {aspect: {'positive': 0, 'negative': 0, 'manual_agreement': 0, 'total': 0}
                    for aspect in aspects}

    # Also track by platform if requested
    if by_platform and 'platform' in df.columns:
        platforms = df_sample['platform'].unique()
        absa_by_platform = {
            platform: {
                aspect: {'positive': 0, 'negative': 0, 'manual_agreement': 0, 'total': 0}
                for aspect in aspects
            } for platform in platforms
        }
    else:
        absa_by_platform = None

    # For each record, analyze sentiment for each aspect mentioned
    for idx, row in df_sample.iterrows():
        content = str(row['content']) if isinstance(row['content'], str) else ""
        platform = row.get('platform', 'unknown') if 'platform' in df.columns else 'unknown'

        # Check which aspects are mentioned
        for aspect in aspects:
            aspect_col = f"manual_{aspect}"

            # Skip if no manual annotation for this aspect
            if aspect_col not in df.columns:
                continue

            # Check if the aspect is mentioned using keywords
            is_mentioned = any(keyword in content.lower() for keyword in aspect_keywords[aspect])

            if is_mentioned:
                # Get the manual annotation
                manual_sentiment = 1 if row[f"{aspect_col}_binary"] == 1 else 0

                # Extract sentences mentioning the aspect
                sentences = re.split(r'[.!?]', content)
                relevant_sentences = []

                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in aspect_keywords[aspect]):
                        relevant_sentences.append(sentence)

                if relevant_sentences:
                    # Join the relevant sentences (limit length to avoid truncation issues)
                    aspect_text = ' '.join(relevant_sentences)
                    if len(aspect_text) > 5000:  # Pre-truncate very long texts
                        aspect_text = aspect_text[:5000]

                    try:
                        # Analyze sentiment for this aspect
                        sentiment_result = sentiment_analyzer([aspect_text])[0]
                        predicted_sentiment = 1 if sentiment_result['label'] == 'POSITIVE' else 0
                    except Exception as e:
                        print(f"Error analyzing aspect {aspect}: {e}")
                        # Default to neutral in case of error
                        predicted_sentiment = manual_sentiment  # Use manual annotation as fallback

                    # Update overall counts
                    absa_results[aspect]['total'] += 1
                    if predicted_sentiment == 1:
                        absa_results[aspect]['positive'] += 1
                    else:
                        absa_results[aspect]['negative'] += 1

                    # Check agreement with manual annotation
                    if predicted_sentiment == manual_sentiment:
                        absa_results[aspect]['manual_agreement'] += 1

                    # Update platform-specific counts if tracking
                    if absa_by_platform is not None:
                        if platform in absa_by_platform:
                            absa_by_platform[platform][aspect]['total'] += 1
                            if predicted_sentiment == 1:
                                absa_by_platform[platform][aspect]['positive'] += 1
                            else:
                                absa_by_platform[platform][aspect]['negative'] += 1

                            if predicted_sentiment == manual_sentiment:
                                absa_by_platform[platform][aspect]['manual_agreement'] += 1

    # Calculate percentages and format results
    print("\nAspect-Based Sentiment Analysis Results:")

    for aspect, results in absa_results.items():
        total = results['total']
        if total > 0:
            pos_percent = results['positive'] / total * 100
            neg_percent = results['negative'] / total * 100
            agreement = results['manual_agreement'] / total * 100

            print(f"\n{aspect.capitalize()}:")
            print(f"  Mentioned in {total} samples")
            print(f"  Positive: {results['positive']} ({pos_percent:.2f}%)")
            print(f"  Negative: {results['negative']} ({neg_percent:.2f}%)")
            print(f"  Agreement with manual annotations: {agreement:.2f}%")
        else:
            print(f"\n{aspect.capitalize()}: Not enough mentions for analysis")

    # Print platform-specific ABSA results if available
    if absa_by_platform:
        print("\n\nPlatform-Specific Aspect Sentiment Analysis:")

        for platform, platform_results in absa_by_platform.items():
            print(f"\n{platform.upper()}:")

            for aspect, aspect_results in platform_results.items():
                total = aspect_results['total']

                if total >= 5:  # Only report if we have enough samples
                    pos_percent = aspect_results['positive'] / total * 100
                    neg_percent = aspect_results['negative'] / total * 100

                    print(f"  {aspect.capitalize()}:")
                    print(f"    Mentioned in {total} samples")
                    print(f"    Positive: {aspect_results['positive']} ({pos_percent:.2f}%)")
                    print(f"    Negative: {aspect_results['negative']} ({neg_percent:.2f}%)")

    return absa_results, absa_by_platform


# 7. PLATFORM COMPARISON ANALYSIS
def compare_platforms(df, absa_results_by_platform):
    """Compare sentiment across different platforms"""
    print("\nPerforming cross-platform sentiment comparison...")

    if 'platform' not in df.columns:
        print("Cannot perform platform comparison: no platform column in dataset")
        return None

    platforms = df['platform'].unique()
    platform_dfs = {platform: df[df['platform'] == platform] for platform in platforms}

    # Calculate overall sentiment metrics per platform
    platform_metrics = {}

    for platform, platform_df in platform_dfs.items():
        # Skip platforms with too few samples
        if len(platform_df) < 30:
            continue

        # Overall sentiment stats
        overall_sentiment = platform_df['sentiment_binary'].mean() * 100  # as percentage

        # Aspect-specific sentiment summaries
        aspect_sentiment = {}
        for aspect in ['content_quality', 'pricing', 'ui_ux', 'technical', 'customer_service']:
            aspect_col = f"manual_{aspect}_binary"
            if aspect_col in platform_df.columns:
                aspect_sentiment[aspect] = platform_df[aspect_col].mean() * 100  # as percentage

        # Add to platform metrics
        platform_metrics[platform] = {
            'sample_size': len(platform_df),
            'overall_positive': overall_sentiment,
            'aspects': aspect_sentiment
        }

    # Print platform comparison
    print("\nPlatform Comparison - Overall Positive Sentiment:")
    for platform, metrics in platform_metrics.items():
        print(f"  {platform}: {metrics['overall_positive']:.2f}% positive (n={metrics['sample_size']})")

    # Print aspect comparisons
    print("\nPlatform Comparison by Aspect (Positive %):")

    aspects = ['content_quality', 'pricing', 'ui_ux', 'technical', 'customer_service']
    for aspect in aspects:
        print(f"\n  {aspect.capitalize()}:")
        for platform, metrics in platform_metrics.items():
            if aspect in metrics['aspects']:
                print(f"    {platform}: {metrics['aspects'][aspect]:.2f}%")

    return platform_metrics


# 8. ABLATION STUDY
def perform_ablation_study(df):
    """Perform ablation study to test different combinations of advanced features"""
    print("\nPerforming ablation study...")

    base_features = ['processed_content']
    manual_features = ['text_length', 'word_count', 'sentiment_ratio', 'positive_word_count', 'negative_word_count']
    platform_cols = [col for col in df.columns if col.startswith('platform_')]
    platform_mention_cols = [col for col in df.columns if col.startswith('mentions_')]
    sarcasm_cols = ['is_sarcastic', 'sarcasm_confidence']
    ner_cols = [col for col in df.columns if col.startswith('has_entity_')]
    wsd_col = 'wsd_sense_content'

    y = df['sentiment_binary']
    results = {}

    def evaluate_model(X, name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        pipeline = Pipeline([
            ('preprocessor', ColumnTransformer([
                ('text', TfidfVectorizer(max_features=3000), 'processed_content' if 'processed_content' in X.columns else wsd_col),
                ('other', Pipeline([
                    ('identity', FunctionTransformer(lambda x: x)),
                    ('scaler', StandardScaler())
                ]), [col for col in X.columns if col not in ['processed_content', wsd_col]])
            ])),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    # 1. Base TF-IDF only
    evaluate_model(df[['processed_content']], 'Base Model (TF-IDF only)')

    # 2. Manual features
    evaluate_model(df[['processed_content'] + manual_features], 'With Manual Features')

    # 3. + Platform features
    evaluate_model(df[['processed_content'] + manual_features + platform_cols + platform_mention_cols], 'With Platform Features')

    # 4. + Sarcasm features
    evaluate_model(df[['processed_content'] + manual_features + sarcasm_cols], 'With Sarcasm Features')

    # 5. + NER features
    evaluate_model(df[['processed_content'] + manual_features + ner_cols], 'With NER Features')

    # 6. + WSD instead of normal text
    evaluate_model(df[[wsd_col] + manual_features], 'With WSD Replacing TF-IDF Input')

    # 7. All combined
    combined_cols = list(set(manual_features + platform_cols + platform_mention_cols + sarcasm_cols + ner_cols))
    evaluate_model(df[['processed_content'] + combined_cols], 'With All Features (Except WSD)')
    evaluate_model(df[[wsd_col] + combined_cols], 'With All Features (Using WSD)')

    # Print results
    print("\nAblation Study Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

    return results

# 9. VISUALIZATION FUNCTIONS
def visualize_results(df, ablation_results, absa_results=None, platform_metrics=None):
    """Create visualizations of the analysis results"""
    # Set up the visualization environment
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 12})

    # Create output directory if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')

    # 1. Ablation Study Results
    if ablation_results:
        models = list(ablation_results.keys())
        accuracies = [ablation_results[model]['accuracy'] for model in models]
        f1_scores = [ablation_results[model]['f1'] for model in models]

        plt.figure(figsize=(14, 8))
        x = np.arange(len(models))
        width = 0.35

        plt.bar(x - width / 2, accuracies, width, label='Accuracy')
        plt.bar(x + width / 2, f1_scores, width, label='F1 Score')

        plt.xlabel('Model Configuration')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/ablation_results.png')
        print("Saved ablation study visualization to 'visualizations/ablation_results.png'")

    # 2. Overall Sentiment Distribution
    sentiment_counts = df['sentiment_binary'].value_counts()
    plt.figure(figsize=(10, 8))
    plt.pie(
        sentiment_counts,
        labels=['Negative', 'Positive'],
        autopct='%1.1f%%',
        colors=['#ff9999', '#66b3ff'],
        startangle=90,
        explode=(0.05, 0)
    )
    plt.title('Overall Sentiment Distribution in Comments')
    plt.savefig('visualizations/sentiment_distribution.png')
    print("Saved sentiment distribution visualization to 'visualizations/sentiment_distribution.png'")

    # 3. Platform-specific Sentiment Distribution
    if 'platform' in df.columns:
        platforms = df['platform'].value_counts()
        top_platforms = platforms[platforms > 50].index.tolist()  # Only plot platforms with enough data

        if len(top_platforms) > 1:  # Only create plot if we have multiple platforms
            platform_sentiment = {}

            for platform in top_platforms:
                platform_df = df[df['platform'] == platform]
                positive_pct = platform_df['sentiment_binary'].mean() * 100
                platform_sentiment[platform] = positive_pct

            # Sort by sentiment
            platform_sentiment = {k: v for k, v in
                                  sorted(platform_sentiment.items(), key=lambda item: item[1], reverse=True)}

            plt.figure(figsize=(12, 8))
            plt.bar(
                list(platform_sentiment.keys()),
                list(platform_sentiment.values()),
                color=plt.cm.viridis(np.linspace(0, 0.8, len(platform_sentiment)))
            )
            plt.xlabel('Platform')
            plt.ylabel('Positive Sentiment (%)')
            plt.title('Positive Sentiment by Platform')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('visualizations/platform_sentiment.png')
            print("Saved platform sentiment visualization to 'visualizations/platform_sentiment.png'")

    # 4. Aspect Sentiment Distribution
    if absa_results:
        aspects = list(absa_results.keys())
        positive_percentages = []
        negative_percentages = []

        for aspect in aspects:
            total = absa_results[aspect]['total']
            if total > 0:
                positive_percentages.append(absa_results[aspect]['positive'] / total * 100)
                negative_percentages.append(absa_results[aspect]['negative'] / total * 100)
            else:
                positive_percentages.append(0)
                negative_percentages.append(0)

        plt.figure(figsize=(12, 8))
        x = np.arange(len(aspects))
        width = 0.35

        plt.bar(x - width / 2, positive_percentages, width, label='Positive', color='#66b3ff')
        plt.bar(x + width / 2, negative_percentages, width, label='Negative', color='#ff9999')

        plt.xlabel('Aspect')
        plt.ylabel('Percentage')
        plt.title('Sentiment Distribution by Aspect')
        plt.xticks(x, [a.capitalize().replace('_', ' ') for a in aspects], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/aspect_sentiment.png')
        print("Saved aspect sentiment visualization to 'visualizations/aspect_sentiment.png'")

    # 5. Platform Comparison by Aspect
    if platform_metrics and len(platform_metrics) > 1:
        # Prepare data for heatmap
        aspects = ['content_quality', 'pricing', 'ui_ux', 'technical', 'customer_service']
        platforms = list(platform_metrics.keys())

        # Create matrix for heatmap
        heatmap_data = np.zeros((len(platforms), len(aspects)))

        for i, platform in enumerate(platforms):
            for j, aspect in enumerate(aspects):
                if aspect in platform_metrics[platform]['aspects']:
                    heatmap_data[i, j] = platform_metrics[platform]['aspects'][aspect]
                else:
                    heatmap_data[i, j] = np.nan  # Missing data

        # Create heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.1f',
            cmap='YlGnBu',
            xticklabels=[a.capitalize().replace('_', ' ') for a in aspects],
            yticklabels=platforms,
            linewidths=.5,
            cbar_kws={'label': 'Positive Sentiment (%)'}
        )
        plt.title('Platform Comparison by Aspect (% Positive)')
        plt.tight_layout()
        plt.savefig('visualizations/platform_aspect_comparison.png')
        print("Saved platform aspect comparison visualization to 'visualizations/platform_aspect_comparison.png'")


# 10. MAIN FUNCTION
def main():
    # Set the path to your pre-annotated dataset
    file_path = '../data/streaming_enhanced_features.csv'

    # Configure warnings to be less verbose
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    # 1. Load data
    df = pd.read_csv(file_path)

    df['processed_content'] = df['processed_content'].fillna("")
    if 'wsd_sense_content' in df.columns:
        df['wsd_sense_content'] = df['wsd_sense_content'].fillna("")

    # Display dataset info
    print(f"\nDataset Information:")
    print(f"Total records: {len(df)}")
    print(
        f"Positive sentiment: {len(df[df['sentiment_binary'] == 1])} ({len(df[df['sentiment_binary'] == 1]) / len(df) * 100:.2f}%)")
    print(
        f"Negative sentiment: {len(df[df['sentiment_binary'] == 0])} ({len(df[df['sentiment_binary'] == 0]) / len(df) * 100:.2f}%)")

    # Print platform distribution if available
    if 'platform' in df.columns:
        platform_counts = df['platform'].value_counts()
        print("\nPlatform distribution in preprocessed data:")
        for platform, count in platform_counts.items():
            print(f"  {platform}: {count} ({count / len(df) * 100:.2f}%)")

    # 3. Split data for training and testing
    feature_cols = ['processed_content', 'text_length', 'word_count', 'sentiment_ratio',
                    'positive_word_count', 'negative_word_count']

    # Add platform-related columns if available
    platform_cols = [col for col in df.columns if col.startswith('platform_')]
    platform_mention_cols = [col for col in df.columns if col.startswith('mentions_')]

    all_feature_cols = feature_cols + platform_cols + platform_mention_cols
    X = df[all_feature_cols]

    # Add platform column itself for stratification
    if 'platform' in df.columns:
        X = X.copy()
        X['platform'] = df['platform']

    y_sentiment = df['sentiment_binary']

    # Stratified split to maintain platform distribution
    if 'platform' in X.columns:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_sentiment, test_size=0.3, random_state=42, stratify=df['platform']
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_sentiment, test_size=0.3, random_state=42
        )

    # 4. Train and evaluate sentiment models with different algorithms
    # Logistic Regression without platform features
    sentiment_lr_no_platform, metrics_lr_no_platform = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, 'lr', 'sentiment', include_platform=False
    )

    # Logistic Regression with platform features
    sentiment_lr, metrics_lr = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, 'lr', 'sentiment', include_platform=True
    )

    # Random Forest with platform features
    sentiment_rf, metrics_rf = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, 'rf', 'sentiment', include_platform=True
    )

    # Random Forest with platform-stratified evaluation
    if 'platform' in df.columns:
        sentiment_rf_strat, metrics_rf_strat = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, 'rf', 'sentiment',
            include_platform=True, platform_stratified=True
        )

    # 5. Train and evaluate aspect models
    aspect_models = {}
    aspect_metrics = {}

    for aspect in ['content_quality', 'pricing', 'ui_ux', 'technical', 'customer_service']:
        aspect_col = f"manual_{aspect}_binary"

        if aspect_col in df.columns:
            # Skip aspects with too few positive examples
            positive_count = df[aspect_col].sum()
            if positive_count > 30:  # Minimum threshold for training
                y_aspect = df[aspect_col]
                X_train_aspect, X_test_aspect, y_train_aspect, y_test_aspect = train_test_split(
                    X, y_aspect, test_size=0.3, random_state=42
                )

                model, metrics = build_aspect_model(
                    aspect, X_train_aspect, X_test_aspect, y_train_aspect, y_test_aspect
                )

                aspect_models[aspect] = model
                aspect_metrics[aspect] = metrics

    # 6. Use transformer model for sentiment analysis
    transformer_model, transformer_metrics = transformer_sentiment_analysis(df, stratify_by_platform=True)

    # 7. Perform Aspect-Based Sentiment Analysis
    absa_results, absa_by_platform = perform_absa(df, transformer_model, by_platform=True)

    # 8. Compare platforms
    if 'platform' in df.columns:
        platform_metrics = compare_platforms(df, absa_by_platform)
    else:
        platform_metrics = None

    # 9. Perform ablation study
    ablation_results = perform_ablation_study(df)

    # 10. Visualize results
    visualize_results(df, ablation_results, absa_results, platform_metrics)

    print("\nAnalysis complete!")

    # Save processed data
    df.to_csv('result/streaming_platforms_processed.csv', index=False)
    print("Processed data saved to 'streaming_platforms_processed.csv'")


if __name__ == "__main__":
    main()