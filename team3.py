import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from pathlib import Path
from collections import Counter
import numpy as np
from typing import List, Dict, Tuple, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from plotly.offline import plot
import plotly.graph_objs as go
import gensim
import random
import scipy.stats as st
import pandas as pd
from pprint import pprint

print("Loading global variable")
stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()
google = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300-SLIM.bin", binary=True)
vocab_google = set(google.vocab.keys())


# Data Preparation
def news_20newsgroups():
    category_names = dict([(i, w) for i, w in enumerate(fetch_20newsgroups().target_names)])
    text, category_num = fetch_20newsgroups(shuffle=False, remove=('headers', 'footers', 'quotes'), return_X_y=True)
    data = [(text[i], category_names[category_num[i]]) for i in range(len(text))]
    return data


def news_bbc(bbc_dir: str = 'bbc'):
    data = []
    for category in ['entertainment', 'business', 'sport', 'politics', 'tech']:
        d_path = Path(f'{bbc_dir}/{category}')
        for f_path in d_path.iterdir():
            if f_path.is_file():
                try:
                    with f_path.open() as f:
                        text = f.read()
                    data.append((text, category))
                except:
                    print(f_path)
    return data


print("Load News Dataset")
data_raw = news_bbc()
__small_dataset = False
if __small_dataset:
    data_raw = random.sample(data_raw, min(500, len(data_raw)))


# Preprocessing (lemmatization)
def preprocessing_simple(text: str) -> str:
    # 37 / sec
    def simplify_pos(tag: str) -> str:
        if tag.startswith('NN'):
            return 'n'
        elif tag.startswith('VB'):
            return 'v'
        elif tag.startswith('RB'):
            return 'r'
        elif tag.startswith('JJ'):
            return 'a'
        else:
            return ''

    lemmas = [lemmatizer.lemmatize(w, simplify_pos(p)).lower() if simplify_pos(p) else w.lower() for w, p in
              nltk.pos_tag(nltk.word_tokenize(text.strip()))]
    words = [w for w in lemmas if w not in stopwords and sum([c.isalpha() for c in w]) >= 3]
    return ' '.join(words)


# Vectorization (TF-IDF)
print("Preprocessing")
texts = [preprocessing_simple(t) for t, c in data_raw]
categories = [c for t, c in data_raw]
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, ngram_range=(1, 2), stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(texts)


# Visualization (t-SNE)
def visualize_vector_category(tfidf, categories: List[str], n_components: int = 3, filename: str = 'plotly.html'):
    tsne = TSNE(n_components=n_components)
    X = tsne.fit_transform(tfidf)
    cat_to_num = {s: i for i, s in enumerate(set(categories))}
    t = np.array([cat_to_num[c] for c in categories])
    color = (t - t.min()) / (t.max() - t.min())
    if n_components == 3:
        scatter = go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            text=['point #{}'.format(i) for i in range(X.shape[0])],
            mode='markers',
            marker=dict(
                size=3,
                color=color,
                colorscale='Jet',
                line=dict(width=0.0),
                opacity=0.8
            )
        )
    elif n_components == 2:
        scatter = go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            text=['point #{}'.format(i) for i in range(X.shape[0])],
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                colorscale='Jet',
                line=dict(width=0.0),
                opacity=0.8
            )
        )
    else:
        scatter = None
    fig = go.Figure(data=[scatter])
    plot(fig, filename=filename, auto_open=False)


print("Visualization")
visualize_vector_category(tfidf, categories, n_components=3, filename='plotly_tfidf_3d.html')  # 30 sec
visualize_vector_category(tfidf, categories, n_components=2, filename='plotly_tfidf_2d.html')  # 20 sec


# Cosine Similarity
def similar_texts(query: str, texts: List[str], tfidf_vectorizer):
    texts_without_query = list(set(texts) - {query})
    tfidf = tfidf_vectorizer.transform(texts_without_query)
    tfidf_query = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(tfidf_query, tfidf).flatten()  # TODO: other similarity algorithm?
    related_texts_indices = similarities.argsort()[::-1]
    sim_texts = [texts[i] for i in related_texts_indices]
    sim_scores = similarities[related_texts_indices]
    return sim_texts, sim_scores


# Advanced Preprocessing
def preprocessing_advanced(text: str) -> str:
    # nltk.help.upenn_tagset()
    # TODO: coreference resolution, 'and' to hypernym, ...
    words = [w for w, p in nltk.pos_tag(nltk.word_tokenize(text.strip()))
             if p[0] in 'FJNRV' and w.lower() not in stopwords and len(w) >= 3 and "'" not in w]
    keywords = [w for w in words if w in vocab_google or w.lower() in vocab_google]
    keywords2 = [w if w in vocab_google else w.lower() for w in keywords]
    return ' '.join(keywords2)


# Keyword-Weight Extraction
def keyword_weight(text: str) -> List[Tuple[str, float]]:
    # TODO: (nouns 5, verbs 5)
    keywords = preprocessing_advanced(text).split()
    counter = Counter(keywords)
    word_freq = counter.most_common()[:10]
    word_freq_normalized = [(w, f / sum([f for w, f in word_freq])) for w, f in word_freq]
    return word_freq_normalized


# Vectorizer
def keyword_vectorizer(texts: List[str]) -> np.ndarray:
    def kw_vector(text: str) -> np.ndarray:
        word_freq = keyword_weight(text)
        vectors = [google.word_vec(w) * f for w, f in word_freq]
        vector = sum(vectors)
        return vector / np.linalg.norm(vector)

    # google.similar_by_vector(keyword_vectorizer(texts[100]))
    return np.vstack([kw_vector(t) for t in texts])


# Visualization
print("Vectorizing")
kwvec = keyword_vectorizer([t for t, c in data_raw])  # 60 sec
categories = [c for t, c in data_raw]
print("Visualization")
visualize_vector_category(kwvec, categories, n_components=3, filename='plotly_kwvec_3d.html')  # 30 sec
visualize_vector_category(kwvec, categories, n_components=2, filename='plotly_kwvec_2d.html')  # 20 sec

# Similarity
print("Cosine similarity")
kwsim = cosine_similarity(kwvec, kwvec)
tfsim = cosine_similarity(tfidf, tfidf)

if __small_dataset:
    print("Visualization")
    tfsim_sorted = np.vstack([tfsim[i][tfsim[i].argsort()[::-1]] for i in range(len(tfsim))])
    kwsim_sorted = np.vstack([kwsim[i][tfsim[i].argsort()[::-1]] for i in range(len(tfsim))])
    plot(go.Figure(data=[go.Surface(z=kwsim_sorted)]), filename='plotly_similarity_kw.html', auto_open=False)
    plot(go.Figure(data=[go.Surface(z=tfsim_sorted)]), filename='plotly_similarity_tf.html', auto_open=False)


# Quantitative Evaluation (nDCG)
def dcg(rel_list: List[float]) -> float:
    return sum([rel / np.log2((i + 1) + 1) for i, rel in enumerate(rel_list)])


def ndcg(rel_list: np.ndarray, ideal_list: np.ndarray) -> float:
    return dcg(ideal_list[rel_list.argsort()[::-1]]) / dcg(ideal_list[ideal_list.argsort()[::-1]])


print('nDCG 95% CI')
scores = np.array([ndcg(kwsim[i], tfsim[i]) for i in range(len(tfsim))])
ci = st.t.interval(0.95, len(scores) - 1, loc=np.mean(scores), scale=st.sem(scores))
print(ci)

__plan = """
Implementing 'Related Articles' Feature for News Corpus
1. data preparation: BBC 

2. simple preprocessing: NLTK
3. vectorization (unigram + bigram) (TF-IDF): sklearn
3.1 visualization -> verification (t-SNE): sklearn
4. cosine similarity: sklearn

2. advanced preprocessing (coreference resolution, hypernym, pos_tag, np_chunking->주어...): NLTK (+ allennlp, spaCy)
3. keyword-weight extraction (NER?, ...) (pseudo vectorization): NLTK (+ allennlp, spaCy)
3.1 visualization -> qualitative evaluation (Word2Vec -> t-SNE): gensim, sklearn
4. words similarity: NLTK

5. quantitative evaluation: score keywords_similarity by tfidf_cosine_similarity
"""
__reference = """
BBC News
http://mlg.ucd.ie/datasets/bbc.html

TF-IDF(Term Frequency-Inverse Document Frequency)
https://wikidocs.net/31698

코사인 유사도(Cosine Similarity)
https://wikidocs.net/24603

t-Stochastic Neighbor Embedding (t-SNE) 와 perplexity
https://lovit.github.io/nlp/representation/2018/09/28/tsne/

Plotly 를 이용한 3D scatter plot
https://lovit.github.io/visualization/2018/04/26/plotly_3d_scatterplot/

Google's trained Word2Vec model in Python
https://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/

gensim - tutorial - word2vec - GoogleNews
https://frhyme.github.io/python-libs/gensim0_word2vec_1google_model/
https://github.com/eyaler/word2vec-slim/raw/master/GoogleNews-vectors-negative300-SLIM.bin.gz

Topic extraction with Non-negative Matrix Factorization and Latent Dirichlet Allocation
https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

Discounted cumulative gain (DCG) is a measure of ranking quality
https://en.m.wikipedia.org/wiki/Discounted_cumulative_gain

Topic Modeling with Gensim (Python)
https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
"""
__presentation = """
max 4 minutes (plus around 1 minute for Q/A)
500 words + 50*3 words
Problem Statement
Technical Approach
Results and Conclusion
Our Works
"""
