import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from gensim.models import doc2vec
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from nltk.util import ngrams



class Embeddings(object):

    def lsa(document, n_components, max_df=1.0, min_df=1, max_features=None, ngram_range=(1, 1)):
        ''' Get LSA embeddings from text document. '''
        
        # initialize and fit tfidf model
        tf = TfidfVectorizer(analyzer='word',
                             max_df=max_df,
                             min_df=min_df,
                             max_features=max_features,
                             ngram_range=ngram_range,
                             stop_words='english')
        tfidf_matrix = tf.fit_transform(document)

        # initialize and fit svd model
        # initialize and train svd model
        tsvd = TruncatedSVD(n_components=n_components)
        svd_matrix = tsvd.fit_transform(tfidf_matrix)

        # transform and return input data
        lsa_matrix = pd.DataFrame(svd_matrix)
        lsa_matrix.columns = ["lsa_" + str(s) for s in np.arange(0, n_components)]
        return lsa_matrix, tf, tsvd
        

    def tfidf_vectorizer(document, max_df=1.0, min_df=1, max_features=None, ngram_range=(1, 1)):
        '''Get tfidf matrix from text document.'''

        # initialize and fit model
        tf = TfidfVectorizer(analyzer='word',
                             max_df=max_df,
                             min_df=min_df,
                             max_features=max_features,
                             ngram_range=ngram_range,
                             stop_words='english')
        tf.fit(document)

        # transform and return input data
        tfidf_matrix = tf.transform(document)
        tfidf_matrix = pd.DataFrame(tfidf_matrix.toarray())
        tfidf_matrix.columns = tf.get_feature_names()

        return tfidf_matrix, tf
    

    def count_vectorizer(document, max_df=1.0, min_df=1, max_features=None, ngram_range=(1, 1)):
        '''Get count matrix from text document. 
        Can also be used to one hot encode categorical data'''
        
        # initialize and fit model
        cv = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features, ngram_range=ngram_range)
        cv = cv.fit(document)

        # transform and return input data
        count_matrix = cv.transform(document)
        count_matrix = pd.DataFrame(count_matrix.toarray())
        count_matrix.columns = cv.get_feature_names()

        return count_matrix, cv

    def pca(X, n_components):
        '''Perform PCA on feature matrix X. 
        Can be used for feature engineering, dimensionality reduction, smoothing, or creating plot axes.'''

        # initialize and fit model
        pca = PCA(n_components=n_components)
        pca = pca.fit(X)

        # transform and return input data
        pca_matrix = pca.transform(X)
        pca_matrix = pd.DataFrame(pca_matrix)
        pca_matrix.columns = ["pca_" + str(s) for s in np.arange(0, n_components)]

        return pca_matrix, pca


    def svd(X, n_components):
        '''Perform SVD on feature matrix X. 
        Can be used for feature engineering, dimensionality reduction, smoothing, or creating plot axes.'''

        # initialize and train svd model
        tsvd = TruncatedSVD(n_components=n_components)
        tsvd = tsvd.fit(X)

        # transform and return input data
        latent_matrix = tsvd.transform(X)
        latent_matrix = pd.DataFrame(latent_matrix)
        latent_matrix.columns = ["svd_" + str(s) for s in np.arange(0, n_components)]

        return latent_matrix, tsvd
    

    def doc_to_vec(document, vector_size, min_count=0, epochs=3, seed=0, window=3, dm=1):
        '''Use doc2vec to create document embeddings from text document.'''

        # format data
        doc_list = []
        for i in range(len(document)):
            mystr = document[i]
            doc_list.append(re.sub("[^\w]", " ",  mystr).split())   
        formatted_documents = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(doc_list)]

        # initialize and train doc2vec model
        model = doc2vec.Doc2Vec(vector_size=vector_size,
                                min_count=min_count,
                                epochs=epochs,
                                seed=seed,
                                window=window,
                                dm=dm)
        model.build_vocab(formatted_documents)
        model.train(formatted_documents, total_examples=model.corpus_count, epochs=model.epochs)

        # transform and return input data
        docvec_matrix = pd.DataFrame(model.docvecs.vectors_docs)
        docvec_matrix.columns = ["docvec_" + str(s) for s in np.arange(0, vector_size)]

        return docvec_matrix, model

    def lda(document, ntopics, n=2):
        '''Use LDA to create document embeddings'''

        # make list of docs  with n-grams
        docs = document.str.split()
        for i in range(len(docs)):
            docs[i] = docs[i] + ["_".join(w) for w in ngrams(docs[i], n)]

        # make dictionary of bow document terms
        dictionary = corpora.Dictionary(docs)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]

        # initialize and train lda model
        ldamodel = LdaModel(corpus=doc_term_matrix,
                            id2word=dictionary,
                            num_topics=ntopics, 
                            random_state=0)

        # transform and return input data
        lda_matrix = pd.DataFrame()
        for doc in docs:
            bow = ldamodel.id2word.doc2bow(doc)
            pred = np.round(ldamodel.get_document_topics(bow=bow,  minimum_probability=0), 4)
            pred = pd.DataFrame(pred)
            lda_matrix = lda_matrix.append(pred[1])
        lda_matrix.reset_index(drop=True, inplace=True)
        lda_matrix.columns = ["lda_" + str(s) for s in np.arange(0, ntopics)]

        return lda_matrix, ldamodel
    

class DataCleaning(object):
    
    def stem_words(text):
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        return text

    def make_lower_case(text):
        return text.lower()

    def remove_stop_words(text):
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
        return text

    def remove_punctuation(text):
        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(text)
        text = " ".join(text)
        return text

    def remove_emails(text):
        string_no_emails = re.sub("\S*@\S*\s?", "", text)
        return (string_no_emails)

    def remove_numbers(text):
        string_no_numbers = re.sub("\d+", "", text)
        return (string_no_numbers)