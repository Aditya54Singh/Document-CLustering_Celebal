
# 📚 Topic Modeling on 20 Newsgroups Dataset

This project is a professional-grade Streamlit application that performs **topic modeling** using **clustering algorithms** like **K-means** and **Latent Dirichlet Allocation (LDA)**. It allows users to explore hidden topics within documents of the [20 Newsgroups dataset](http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) in an interactive and visual manner.

---

## 🚀 Features

- ✅ Choose between **K-means** and **LDA** clustering algorithms.
- ✅ Customize the number of **topics/clusters**.
- ✅ Select specific **categories** from the dataset.
- ✅ View **top terms** for each topic.
- ✅ Interactive **word clouds**.
- ✅ Download results as **CSV**.
- ✅ Optimized using **Streamlit caching** for performance.

---

## 📂 Project Structure

```
├── app.py                  # Main Streamlit application
├── requirements.txt        # List of dependencies
├── README.md               # This file
├── assets/
│   └── app_preview.png     # Optional: Image preview for README
└── output/
    └── Kmeans_topics.csv   # Sample output files (if any)
```

---

## 🧠 Algorithms Used

### 1. **K-means Clustering**
- Unsupervised clustering based on TF-IDF vector space.
- Displays top keywords from each cluster's centroid.

### 2. **Latent Dirichlet Allocation (LDA)**
- Probabilistic generative model for topic extraction.
- Topics are interpreted via highest-weighted terms.

---

## 📝 Dataset

- **Source**: [20 Newsgroups](http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups)
- Consists of ~20,000 documents across 20 categories.
- Includes topics like tech, religion, politics, sports, etc.
- Categories can be filtered through the app UI.

---

## 🔧 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/topic-modeling-streamlit.git
cd topic-modeling-streamlit

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Run the Streamlit app
streamlit run app.py
```

---

## 📦 Requirements

See `requirements.txt` for the full list. Key packages include:

- `streamlit`
- `scikit-learn`
- `nltk`
- `wordcloud`
- `matplotlib`
- `pandas`

Ensure NLTK downloads:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 📊 How It Works

1. **Load Dataset**: Dynamically fetches and filters the newsgroups.
2. **Preprocessing**:
   - Lowercase conversion
   - Stopword removal
   - Lemmatization
3. **Vectorization**:
   - TF-IDF used to convert text into numeric features.
4. **Clustering**:
   - User selects either K-means or LDA with topic count.
5. **Visualization**:
   - Word clouds and keyword lists for each topic.
6. **Download**:
   - Topics can be exported as a `.csv` file.

---

## 📈 Use Cases

- NLP-based document analysis
- Exploratory topic modeling
- Unsupervised classification
- Visual understanding of large text corpora
- Academic/research applications

---

## 🙋‍♂️ Author

**Aditya Kumar Singh**  
_Data Science Intern | Research Enthusiast_  

---


