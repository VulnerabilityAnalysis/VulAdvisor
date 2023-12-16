from sklearn.feature_extraction.text import TfidfVectorizer


def get_vector(corpus):
    vectorizer = TfidfVectorizer(lowercase=False)
    vectorizer.fit(corpus)
    return vectorizer


def get_top_terms(vectorizer, queries, k):
    # Transform the corpus into a matrix of TF-IDF features
    tfidf_matrix = vectorizer.transform(queries)

    # Get the feature names (keywords)
    feature_names = vectorizer.get_feature_names_out()

    # Print the top 5 keywords for each document
    for i in range(len(queries)):
        print(f"Keywords for document {i+1}:")
        feature_index = tfidf_matrix[i,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
        sorted_tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        key_terms = []
        for j in range(k):
            try:
                key_terms.append(feature_names[sorted_tfidf_scores[j][0]])
            except IndexError:
                print(queries[i])
                break
            print(f"  {feature_names[sorted_tfidf_scores[j][0]]} ({sorted_tfidf_scores[j][1]:.2f})")
        queries[i] = f"# {' '.join(key_terms)} # "
    return queries


def process(dataset, vectorizer=None):
    with open(f'data/{dataset}.code') as fp:
        tokenized_code = fp.readlines()

    terms = tokenized_code
    with open(f"data/{dataset}.term", 'w') as fw:
        if not vectorizer:
            vectorizer = get_vector(terms)
        terms_code = get_top_terms(vectorizer, terms, 5)
        for item in terms_code:
            fw.write(item+'\n')
    return vectorizer


vec = process('train')
process('valid', vec)
process('test', vec)