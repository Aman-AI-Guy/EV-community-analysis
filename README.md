# Project: Analyzing Electric Vehicle Discussions on Reddit

## Introduction

This project aims to collect, process, and analyze discussions related to electric vehicles (EVs) from various subreddits using Natural Language Processing (NLP) techniques, network analysis, and data visualization. The goal is to gain insights into public opinions, key topics, user interactions, and trends surrounding EVs on the Reddit platform. The project is divided into several distinct phases, from data extraction to advanced analysis like topic modeling and network analysis.

## Phase 1: Data Extraction from Reddit

*   **Objective:** Extract relevant posts about electric vehicles from specified subreddits.
*   **Methodology:**
    *   Utilized the `asyncpraw` library for asynchronous interaction with the Reddit API.
    *   Targeted subreddits: `electricvehicles`, `teslamotors`, `EVnews`.
    *   Searched for posts containing the query "electric vehicle".
    *   Extracted key information for each post: `id`, `created_utc`, `title`, `selftext`, `score`, `num_comments`, and `author`.
    *   Limited extraction to approximately 500 posts per subreddit for this analysis.
*   **Output:** A CSV file (`ev_posts.csv`) containing the raw extracted post data.
*   **Libraries:** `asyncpraw`, `pandas`, `asyncio`, `nest_asyncio`.

## Phase 2: Data Preprocessing

*   **Objective:** Clean and prepare the extracted text data for subsequent analysis.
*   **Methodology:**
    *   Converted the `created_utc` timestamp (Unix epoch time) to a standard datetime format.
    *   Combined the `title` and `selftext` fields into a single `text` column for unified analysis.
    *   Performed text cleaning:
        *   Removed URLs using regular expressions (`re`).
        *   Removed special characters and punctuation, keeping only alphabetic characters and spaces.
        *   Tokenized the text using `nltk.word_tokenize`.
        *   Removed common English stopwords using `nltk.corpus.stopwords`.
    *   Stored the cleaned text in a new `clean_text` column.
*   **Output:** An updated CSV file (`preprocessed_ev_posts.csv`) with added `text` and `clean_text` columns.
*   **Libraries:** `pandas`, `re`, `nltk`.

## Phase 3: Visualization of Data Distribution

*   **Objective:** Gain initial insights into the dataset through visualizations.
*   **Methodology:**
    *   **Temporal Distribution:** Plotted a histogram of post creation dates (`created_utc`) to visualize the distribution of posts over time.
    *   **Word Cloud:** Generated a word cloud from the aggregated `clean_text` to visually identify the most frequent terms in the discussions.
*   **Output:**
    *   Histogram showing post frequency over time.
    *   Word cloud image highlighting common terms.
*   **Libraries:** `matplotlib`, `wordcloud`.

## Phase 4: Information Retrieval

*   **Objective:** Identify posts most relevant to specific search queries using semantic similarity.
*   **Methodology:**
    *   Utilized TF-IDF (Term Frequency-Inverse Document Frequency) vectorization (`sklearn.feature_extraction.text.TfidfVectorizer`) on the `clean_text`.
    *   Defined specific queries (e.g., "electric vehicle", "EV", "Tesla").
    *   Calculated the cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`) between the TF-IDF vector of each query and the vectors of all posts.
    *   Identified and displayed the top 10 posts with the highest similarity scores for each query.
    *   Visualized the top posts for each query using bar plots based on their Reddit score (`seaborn`).
*   **Output:**
    *   Printed lists of top-matching post titles and text for each query.
    *   Bar plots showing the scores of the most relevant posts per query.
*   **Libraries:** `sklearn`, `seaborn`, `matplotlib`, `pandas`.

## Phase 5: Named Entity Recognition (NER)

*   **Objective:** Extract and analyze named entities (like locations, organizations, people) mentioned in the posts.
*   **Methodology:**
    *   Used the `spacy` library with the `en_core_web_sm` model to process the `clean_text`.
    *   Extracted all named entities and their labels (e.g., GPE for Geopolitical Entity, ORG for Organization).
    *   Specifically focused on extracting Geographic Locations (GPE).
    *   Counted the frequency of mentioned locations and all entities.
    *   Visualized the top 10 most frequent locations and top 10 overall named entities using bar charts.
*   **Output:**
    *   DataFrame columns containing extracted locations and all entities for each post.
    *   Frequency counts of entities.
    *   Bar charts visualizing the most common locations and entities.
*   **Libraries:** `spacy`, `pandas`, `matplotlib`.

## Phase 6: Sentiment Analysis

*   **Objective:** Determine the overall sentiment (positive/negative) expressed in the Reddit posts.
*   **Methodology:**
    *   Employed a pre-trained sentiment analysis pipeline from the Hugging Face `transformers` library (model: `distilbert-base-uncased-finetuned-sst-2-english`).
    *   Truncated text using the model's tokenizer (`DistilBertTokenizerFast`) to fit input size limits (max 512 tokens).
    *   Applied the sentiment pipeline to the `truncated_text` to classify each post as 'POSITIVE' or 'NEGATIVE'.
    *   Visualized the overall distribution of sentiments using a countplot (`seaborn`).
    *   Analyzed and plotted sentiment trends over time by grouping data by month.
*   **Output:**
    *   A CSV file (`sentiment_ev_posts.csv`) with sentiment labels added.
    *   A countplot showing the overall sentiment distribution.
    *   A line plot illustrating changes in positive/negative sentiment over months.
*   **Libraries:** `transformers`, `pandas`, `seaborn`, `matplotlib`.

## Phase 7: Topic Modeling

*   **Objective:** Identify latent topics or themes present within the collection of posts.
*   **Methodology:**
    *   Preprocessed the `clean_text` specifically for topic modeling (tokenization, lowercasing, removing stopwords, keeping alphabetic tokens) using `nltk`.
    *   Created a document-term matrix (corpus) and dictionary using `gensim.corpora`.
    *   Trained a Latent Dirichlet Allocation (LDA) model (`gensim.models.LdaModel`) to discover 5 topics.
    *   Displayed the top words associated with each identified topic.
    *   Generated an interactive visualization of the topics and their relationships using `pyLDAvis`.
*   **Output:**
    *   List of keywords defining each topic.
    *   An interactive `pyLDAvis` plot for exploring topics.
*   **Libraries:** `gensim`, `pyLDAvis`, `nltk`, `pandas`.

## Phase 8: Network Analysis

*   **Objective:** Analyze the interaction patterns between users (authors and commenters) and identify influential users and community structures.
*   **Methodology:**
    *   Fetched posts *and* their associated comments using `asyncpraw` (limited scope for feasibility).
    *   Constructed an interaction graph using `networkx`:
        *   Nodes represent Reddit users (post authors and commenters).
        *   Edges represent interactions (a user commenting on a post, or users replying within the same comment thread).
    *   Calculated network centrality measures (Degree, Betweenness, Closeness) to identify potentially influential users.
    *   Applied the Louvain method (`community.community_louvain`) for community detection to find clusters of interacting users.
    *   Visualized the user interaction network graph, optionally colored by detected communities.
*   **Output:**
    *   A network graph (`networkx.Graph`) representing user interactions.
    *   Printed lists of top users based on centrality scores.
    *   Community partition data.
    *   Visualizations of the network graph and its community structure.
*   **Libraries:** `asyncpraw`, `pandas`, `networkx`, `community`, `matplotlib`, `json`.
