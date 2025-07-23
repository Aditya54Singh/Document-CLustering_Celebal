import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Topic Modeling on 20 Newsgroups",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- NLTK Data Download ---
# Download necessary NLTK data in a robust way
try:
    stopwords.words('english')
except LookupError:
    st.info("Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    st.info("Downloading NLTK WordNet...")
    nltk.download('wordnet')


# --- Caching Functions for Performance ---
@st.cache_data
def load_dataset(categories):
    """Loads the 20 Newsgroups dataset, caching the result."""
    data = fetch_20newsgroups(subset='all',
                              categories=categories,
                              shuffle=True,
                              random_state=42,
                              remove=('headers', 'footers', 'quotes'))
    return data

@st.cache_data
def preprocess_text(docs):
    """Preprocesses text data: cleaning, tokenization, lemmatization."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_docs = []
    for doc in docs:
        # Remove non-alphabetic characters and lowercase
        doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
        doc = doc.lower()
        # Tokenize and lemmatize, removing stopwords
        tokens = [lemmatizer.lemmatize(word) for word in doc.split() if word not in stop_words and len(word) > 2]
        processed_docs.append(" ".join(tokens))
    return processed_docs

@st.cache_resource
def vectorize_text(data):
    """Vectorizes text data using TF-IDF, caching the resource."""
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                 max_features=1000,
                                 stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data)
    return tfidf_matrix, vectorizer

# --- Helper function for CSV conversion ---
@st.cache_data
def convert_df_to_csv(df):
  """Converts a DataFrame to a CSV file, cached for performance."""
  return df.to_csv(index=False).encode('utf-8')


# --- Main Application ---

# --- Header ---
st.title("📚 Topic Modeling with Clustering Algorithms")
st.markdown("""
This application performs topic modeling on the **20 Newsgroups dataset**.
You can choose between **K-means** and **Latent Dirichlet Allocation (LDA)** to discover underlying topics in the text corpus.
""")
st.markdown("---")


# --- Sidebar for User Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    st.markdown("Select the algorithm and its parameters.")

    # Algorithm selection
    algorithm = st.selectbox(
        "Choose a Clustering Algorithm",
        ("K-means", "Latent Dirichlet Allocation (LDA)"),
        help="K-means groups documents based on distance, while LDA is a probabilistic model for topic discovery."
    )

    # Number of topics/clusters
    num_topics = st.slider(
        "Select the Number of Topics/Clusters",
        min_value=2, max_value=15, value=5,
        help="How many topics do you want to find in the data?"
    )

    # Dataset category selection
    st.markdown("---")
    st.header("📰 Newsgroup Categories")
    all_categories = fetch_20newsgroups(subset='all').target_names
    selected_categories = st.multiselect(
        "Select categories to analyze (or leave blank for all)",
        all_categories,
        default=['sci.space', 'comp.graphics', 'rec.sport.hockey', 'talk.politics.guns', 'alt.atheism'],
        help="Choosing fewer categories can lead to more distinct topics."
    )
    if not selected_categories:
        selected_categories = None # Use all categories if none are selected


# --- Main Content Area ---
st.header("📊 Results")

# --- Data Loading and Preprocessing ---
with st.spinner("Loading and preprocessing data... This might take a moment."):
    newsgroups_data = load_dataset(selected_categories)
    if not newsgroups_data.data:
        st.warning("No data found for the selected categories. Please select at least one category.")
        st.stop()

    processed_docs = preprocess_text(newsgroups_data.data)
    tfidf_matrix, vectorizer = vectorize_text(processed_docs)
    feature_names = vectorizer.get_feature_names_out()


# --- Model Training and Topic Extraction ---
results_df = None

if algorithm == "K-means":
    st.subheader(f"K-means Clustering Results for {num_topics} Clusters")
    with st.spinner(f"Running K-means with {num_topics} clusters..."):
        # --- K-means Clustering ---
        km = KMeans(n_clusters=num_topics, init='k-means++', max_iter=100, n_init=10, random_state=42)
        km.fit(tfidf_matrix)
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        # --- Display Topics and Prepare for Download ---
        st.markdown("**Top terms per cluster:**")
        topic_results = []
        cols = st.columns(3)
        for i in range(num_topics):
            with cols[i % 3]:
                with st.expander(f"**Cluster {i+1}**", expanded=True):
                    top_terms_list = [feature_names[ind] for ind in order_centroids[i, :15]]
                    top_terms_str = ", ".join(top_terms_list)
                    topic_results.append({'Cluster': f"Cluster {i+1}", 'Top Terms': top_terms_str})

                    st.write(top_terms_str)

                    # --- Word Cloud ---
                    wordcloud_text = " ".join(top_terms_list)
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
        
        results_df = pd.DataFrame(topic_results)


elif algorithm == "Latent Dirichlet Allocation (LDA)":
    st.subheader(f"LDA Results for {num_topics} Topics")
    with st.spinner(f"Running LDA with {num_topics} topics..."):
        # --- LDA ---
        lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10,
                                        learning_method='online',
                                        random_state=42)
        lda.fit(tfidf_matrix)

        # --- Display Topics and Prepare for Download ---
        st.markdown("**Top words per topic:**")
        topic_results = []
        cols = st.columns(3)
        for topic_idx, topic in enumerate(lda.components_):
             with cols[topic_idx % 3]:
                with st.expander(f"**Topic {topic_idx + 1}**", expanded=True):
                    top_words_indices = topic.argsort()[:-15 - 1:-1]
                    top_words = [feature_names[i] for i in top_words_indices]
                    top_words_str = ", ".join(top_words)
                    topic_results.append({'Topic': f"Topic {topic_idx + 1}", 'Top Words': top_words_str})
                    
                    st.write(top_words_str)

                    # --- Word Cloud ---
                    wordcloud_text = " ".join(top_words)
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
        results_df = pd.DataFrame(topic_results)


# --- Download Button ---
if results_df is not None:
    st.markdown("---")
    st.header("⬇️ Download Results")
    
    csv = convert_df_to_csv(results_df)

    st.download_button(
        label="Download topic data as CSV",
        data=csv,
        file_name=f'{algorithm}_topics.csv',
        mime='text/csv',
    )


# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: grey;">
    <p>Created with Streamlit | Dataset: 20 Newsgroups</p>
</div>
""", unsafe_allow_html=True)
