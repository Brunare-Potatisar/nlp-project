# 20180333 Seyoung Song
# rev. 20170818 Sunjoo Yoon

import nltk
from nltk.corpus import reuters
import pandas as pd
from pprint import pprint
from random import randrange
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import string


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

# pprint(similar_docs(reuters_df.fileids[0], reuters_df, cosine_sim)[:10])
# print(doc_similarity(reuters_df.fileids[0], reuters_df.fileids[1], reuters_df, cosine_sim))

# TESTING 1: MANUALLY CHECK RECOMMENDED ARTICLES FOR HUMAN-VERIFIED SIMILARITY WITH ORIGINAL

titlewords = ["lt", "Lt", "ltd", "Ltd", "co", "Co"] # words not capitalised even in title
for i in range(5):
    random_number = randrange(len(reuters_df.fileids)) 
    # get recommendations for five random articles
    ret = similar_docs(reuters_df.fileids[random_number], reuters_df, cosine_sim)
    for j in range(4):
        # get the text of the original article and top three recommendations
        text = reuters.words(ret[j][0])
        # the titles of each article is capitalised, so use that to get the title
        k = 0
        title = ""
        while (text[k].isupper()) or (text[k][0] in string.punctuation) or text[k] in titlewords:
            title = title + text[k].lower() + " "
            k = k + 1
        title = title[:(len(title) - 1)]
        # get the tags of the articles
        tags = reuters.categories(ret[j][0])
        # print the article titles and their tags
        if j == 0:
            print("ORIGINAL ARTICLE:", title, "TAGS:", tags)
        else:
            print("RELATED ARTICLE:", title, "TAGS:", tags)

# TESTING 2: AUTOMATICALLY CHECK RECOMMENDATIONS FOR SHARED TAGS WITH ORIGINAL ARTICLE

score = 0
for i in range(1000):
    # get recommendations for a thousand random articles
    random_number = randrange(len(reuters_df.fileids))
    ret = similar_docs(reuters_df.fileids[random_number], reuters_df, cosine_sim)
    original_tags = reuters.categories(ret[0][0])
    for j in range(3):
        # only consider the top three recommendations for each original article
        tags = reuters.categories(ret[j + 1][0])
        # count how many of them have shared tags with the original article
        flag = 0
        for k in range(len(tags)):
            if tags[k] in original_tags:
                flag = 1
        if flag == 1:
            score = score + 1
# print the result
print("3000 recommendations generated, three each for 1000 randomly-selected articles")
print(score, "of 3000 had at least one shared reuters tag with the original article")