# 20180333 Seyoung Song

import pandas as pd
import nltk
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
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


def cosine_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['text'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)


def doc_similarity(fileid1, fileid2, _data, _cosine_sim):
    idx_to_fileid = dict(_data['fileids'])
    fileid_to_idx = {v: k for k, v in idx_to_fileid.items()}
    idx1 = fileid_to_idx[fileid1]
    idx2 = fileid_to_idx[fileid2]
    return _cosine_sim[idx1][idx2]


def similar_docs(fileid, data, cosine_sim):
    idx_to_fileid = dict(data['fileids'])
    fileid_to_idx = {v: k for k, v in idx_to_fileid.items()}
    idx = fileid_to_idx[fileid]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    return [(idx_to_fileid[i[0]], i[1]) for i in sim_scores]


reuters_df = reuters_dataframe()
cosine_sim = cosine_similarity(reuters_df)

pprint(similar_docs(reuters_df.fileids[0], reuters_df, cosine_sim)[:10])
print(doc_similarity(reuters_df.fileids[0], reuters_df.fileids[1], reuters_df, cosine_sim))
