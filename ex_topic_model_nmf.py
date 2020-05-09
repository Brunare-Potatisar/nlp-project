# 20180333 Seyoung Song
# topic modeling using NMF in sklearn
# NMF is similar to LDA

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk
from nltk.corpus import reuters
import pandas as pd
from pprint import pprint


def reuters_dataframe(n=9160):
    def clean(words):
        stopwords = set(nltk.corpus.stopwords.words('english'))
        words_lower = [w.lower() for w in words]
        return [w for w in words_lower if w not in stopwords and len(w) >= 3]

    def title(words):
        return words[:20]

    fileids = [i for i in reuters.fileids() if len(reuters.categories(i)) == 1][:n]
    df = pd.DataFrame({'text': [' '.join(clean(reuters.words(i))) for i in fileids],
                       'category': [reuters.categories(i)[0] for i in fileids],
                       'title': [' '.join(title(reuters.words(i))) for i in fileids],
                       'fileids': fileids,
                       'words': [reuters.words(i) for i in fileids]})
    return df


def topic_modeling():
    reuters_df = reuters_dataframe()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(reuters_df['text'])
    nmf = NMF(n_components=10, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    topic_keywords = {}
    for topic_idx, topic in enumerate(nmf.components_):
        topic_keywords[topic_idx] = [tfidf_feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    return topic_keywords


topic_kw = topic_modeling()
pprint(topic_kw)
