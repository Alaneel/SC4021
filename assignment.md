# Opinion Search Engine Assignment

## Overview

In this group assignment, you will build an opinion search engine. Given a specific topic of your choice (e.g., cryptocurrencies), your system should enable users to find relevant opinions about any instance of such topic (e.g., bitcoin) and perform sentiment analysis on the results (e.g., opinions about bitcoin are 70% positive and 30% negative).

When selecting a topic, ensure that:
1. You can find enough data about it (avoid topics too niche with only a few hundred data points)
2. The opinions you gather are balanced (avoid topics with only negative or only positive opinions)

For topic ideas, check the project page at https://sentic.net/projects.

You may use available tools/libraries, but your system cannot be just a mashup of existing services. Your final score depends on system development, novelty, and creativity.

**Main tasks and point distribution:**
- Crawling (20 points)
- Indexing (40 points)
- Classification (40 points)

A minimum of 60 points is required to pass the assignment.

## A. Crawling (20 points)

Crawl text data from sources you are interested in and permitted to access (e.g., X API or Reddit API). Requirements:
- At least 10,000 records
- At least 100,000 words
- No duplicates
- Balanced dataset (e.g., equal number of positive and negative entries)

You may use available datasets for training (e.g., popular sentiment benchmarks), but you must crawl and label data for testing.

**Suggested third-party libraries:**
- Jsoup: https://jsoup.org
- Twitter4j: https://twitter4j.org
- Facebook marketing: https://developers.facebook.com/docs/marketing-apis
- Instagram: https://instagram.com/developer
- Amazon: https://github.com/ivanpgs/amazon-crawler
- Tinder: https://gist.github.com/rtt/10403467
- Tik Tok: https://developers.tiktok.com

### Question 1: Explain and provide the following:

1. How you crawled the corpus (e.g., source, keywords, API, library) and stored it
2. What kind of information users might like to retrieve from your crawled corpus (i.e., applications), with sample queries
3. The numbers of records, words, and types (i.e., unique words) in the corpus

## B. Indexing (40 points)

You can implement indexing from scratch or use available tools like Solr+Lucene+Jetty.

- Solr runs as a standalone full-text search server within a servlet container like Jetty
- Solr uses the Lucene search library for text indexing and search
- Solr has REST-like HTTP/XML and JSON APIs for easy integration with any programming language

**Useful documentation:**
- Solr project: https://solr.apache.org
- Solr wiki: https://wiki.apache.org/solr/FrontPage
- Lucene tutorial: https://lucene.apache.org/core/quickstart.html
- Solr with Jetty: https://wiki.apache.org/solr/SolrJetty
- Jetty tutorial: https://jetty.org

You may choose other inverted-index text search engine projects (e.g., Sphinx, Nutch, Lemur) but should not adopt SQL-based solutions for text search.

**User Interface Requirements:**
- Provide a simple but friendly UI for querying
- Can be web-based or mobile app based
- Can use JSP in Java or Django in Python
- A sophisticated UI is not necessary
- Detailed information besides text is allowed (e.g., product images, ratings)

### Question 2: Perform the following tasks:

- Design a simple UI (from scratch or based on an existing one, e.g., Solr UI) for user access
- Write five queries, get their results, and measure the querying speed

### Question 3: Explore innovations for enhancing indexing and ranking.

Explain why they are important to solve specific problems, illustrated with examples. List improvements from your first to last version, including queries that now work due to these improvements.

**Potential innovations include:**
- Timeline search (e.g., search within specific time windows)
- Geo-spatial search (e.g., use map information to refine results)
- Enhanced search (e.g., add histograms, pie charts, word clouds)
- Interactive search (e.g., refine results based on users' relevance feedback)
- Multimodal search (e.g., implement image or video retrieval)
- Multilingual search (e.g., enable retrieval in multiple languages)
- Multifaceted search (e.g., visualize information by different categories)

## Suitcase Model - Natural Language Processing Framework

### Three-Layer Model Overview

Suitcase Model is a comprehensive natural language processing framework organized as a three-layer architecture resembling a suitcase. Each layer builds upon the previous one to process text with increasing levels of sophistication.

#### Syntactics Layer (Foundation)
This initial layer focuses on basic text processing and linguistic structure:
- Microtext normalization - Standardizing text formats and handling irregularities
- Sentence boundary disambiguation - Determining where sentences begin and end
- POS tagging - Identifying parts of speech (nouns, verbs, etc.)
- Text chunking - Grouping words into meaningful phrases
- Lemmatization - Reducing words to their base or dictionary forms

#### Semantics Layer (Middle)
This layer addresses meaning and context:
- Word sense disambiguation - Determining the correct meaning of words with multiple definitions
- Named entity recognition - Identifying and classifying proper nouns
- Concept extraction - Identifying key ideas and themes
- Anaphora resolution - Resolving references (e.g., pronouns to their antecedents)
- Subjectivity detection - Distinguishing objective facts from subjective opinions

#### Pragmatics Layer (Top)
This layer handles higher-level understanding and interpretation:
- Metaphor understanding - Interpreting figurative language
- Sarcasm detection - Recognizing when literal meaning differs from intended meaning
- Personality recognition - Identifying traits or characteristics of the writer/speaker
- Aspect extraction - Identifying specific attributes being discussed
- Downstream task - Applying the processed information to specific applications

The framework shows a sequential flow where output from the syntactics layer feeds into the semantics layer, which then feeds into the pragmatics layer, ultimately supporting various downstream applications.

## C. Classification (40 points)

Choose two or more subtasks from the NLU suitcase model to perform information extraction on your crawled data. For example, you could choose subjectivity detection and polarity detection to:
1. Categorize data as neutral versus opinionated
2. Classify resulting opinionated data as positive versus negative

**Classification approaches:**
- Knowledge-based (e.g., SenticNet)
- Rule-based (e.g., linguistic patterns)
- Machine learning-based (e.g., deep neural networks)
- Hybrid (a combination of any of the above)

**Suggested resources and toolkits:**
- Weka: https://cs.waikato.ac.nz/ml/weka
- Hadoop: https://hadoop.apache.org
- Pylearn2: https://pylearn2.readthedocs.io/en/latest
- SciKit: https://scikit-learn.org
- NLTK: https://nltk.org
- Theano: https://github.com/Theano
- Keras: https://github.com/fchollet/keras
- Tensorflow: https://github.com/tensorflow/tensorflow
- PyTorch: https://pytorch.org
- Huggingface: https://huggingface.co/
- AllenNLP: https://github.com/allenai/allennlp

### Question 4: Perform the following tasks:

- Justify your classification approach choice in relation to the state of the art
- Discuss whether you had to preprocess data and why
- Build an evaluation dataset by manually labeling at least 1,000 records with an inter-annotator agreement of at least 80%
- Provide metrics such as precision, recall, and F-measure on such dataset
- Perform a random accuracy test on the rest of the data and discuss results
- Discuss performance metrics (e.g., speed, scalability) of the system

### Question 5: Explore innovations for enhancing classification.

If you introduce more than one innovation, perform an ablation study to show the contribution of each. For example, if you perform WSD and NER to enhance sentiment analysis, show:
- Accuracy increase when adding only WSD
- Accuracy increase when adding only NER
- Accuracy increase when adding both WSD and NER

Explain why these innovations are important to solve specific problems, illustrated with examples.

**Potential innovations include:**
- Enhanced classification (add another NLU subtask, e.g., sarcasm detection)
- Fine-grained classification (e.g., perform ABSA)
- Hybrid classification (e.g., apply both symbolic and subsymbolic AI)
- Cognitive classification (e.g., use brain-inspired algorithms)
- Multitask classification (e.g., perform two or more NLU tasks jointly)
- Ensemble classification (e.g., use stacked ensemble)

## D. Submission

The submission shall consist of one single PDF file. Add pictures to make your report clearer and easier to read. There is no page limit and no special formatting is required.

**Required components:**
1. Names of the group members on the first page
2. Answers to all the above questions
3. A YouTube link to a video presentation (up to 5 minutes) that includes:
   - Introduction of group members and their roles
   - Explanation of applications and impact of your work
   - Highlights of creative aspects of your work
4. A Dropbox (or similar) link to a compressed file with:
   - Crawled text data
   - Queries and their results
   - Evaluation dataset
   - Automatic classification results
   - Any other data for Questions 3 and 5
5. A Dropbox (or similar) link to a compressed file with:
   - All source codes and libraries
   - README file explaining how to compile and run the source codes