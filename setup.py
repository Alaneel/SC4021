from setuptools import setup, find_packages

setup(
    name="ev_opinion_search",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas>=1.5.3",
        "numpy>=1.24.3",
        "flask>=2.3.2",
        "pysolr>=3.9.0",
        "transformers>=4.28.1",
        "torch>=2.0.1",
        "scikit-learn>=1.2.2",
        "spacy>=3.5.3",
        "gensim>=4.3.1",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "wordcloud>=1.9.2",
        "tqdm>=4.65.0",
        "praw>=7.7.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "ev-crawler=scripts.run_crawler:main",
            "ev-indexer=scripts.run_indexer:main",
            "ev-evaluate=scripts.run_evaluation:main",
            "ev-webapp=scripts.run_webapp:main",
        ],
    },
)