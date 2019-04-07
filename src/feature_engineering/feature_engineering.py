import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle


class Embeddings(object):

    def lsa(self, df, path_to_models, n_components, max_df=1.0, min_df=1, max_features=None, ngram_range=(1, 1)):
        ''' Get LDA embeddings from text data. Trains and saves LSA model.'''
        tfidf_matrix = self.tfidf(df, path_to_models, max_df=1.0, min_df=1, max_features=None, ngram_range=(1, 1))
        lsa_matrix = self.svd(tfidf_matrix, n_components, path_to_models)
        return lsa_matrix
        

    def tfidf(self, df, path_to_models, max_df=1.0, min_df=1, max_features=None, ngram_range=(1, 1)):
        '''Get tfidf matrix from text data. Trains and saves tfidf model.'''

        # initialize and fit model, transform input data
        tf = TfidfVectorizer(analyzer='word',
                             max_df=max_df,
                             min_df=min_df,
                             max_features=max_features,
                             ngram_range=ngram_range,
                             stop_words='english')

        tf.fit(df['document'])

        # save trained model for future use
        pickle.dump(tf, open(path_to_models + "/tfidf_model.pkl", "wb"))

        # transform and return input data
        tfidf_matrix = tf.transform(df['document'])
        tfidf_matrix = pd.DataFrame(tfidf_matrix.toarray())
        tfidf_matrix.columns = tf.get_feature_names()

        return tfidf_matrix

    def count_vectorizor(self, df):
        '''Get word count matrix from text data, Trains and saves cv model.'''
        return df

    def pca(self, df):
        '''Perform pca on feature matrix. Can be used for dimensionality reduction, smoothing, or creating plot axes. Trains and saves pca model.'''
        return df

    def svd(self, df, n_components, path_to_models):
        '''Perform svd on feature matrix. Can be used for dimensionality reduction, smoothing, or creating plot axes. Trains and saves svd model.'''

        # initialize and train svd model
        tsvd = TruncatedSVD(n_components=n_components)
        tsvd = tsvd.fit(df)

        # save trained model for future use
        pickle.dump(tsvd, open(path_to_models + "/svd_model.pkl", "wb"))

        # transform and return input data
        latent_matrix = tsvd.transform(df)
        latent_matrix = pd.DataFrame(latent_matrix)
        latent_matrix.columns = ["svd_" + str(s) for s in np.arange(0, n_components)]

        return latent_matrix

    def doc2vec(self, df):
        '''Use doc2vec to create document embeddings'''
        return df

    def lda(self, df):
        '''Use LDA to create document embeddings'''
        return ds

    def ohe_features(self, df, feature, frequency_threshold):
        '''
        One-hot-encode a categorical feature into binary columns.
        df: pandas data frame with feature to be encoded
        feature: str. feature column name
        frequency_threshold: number of occurrences to threshold feature at.
        '''
        vc = df[feature].value_counts()
        keep_values = vc[vc > frequency_threshold].index.tolist()
        ohe_feature = pd.get_dummies(df[feature])

        feature_names = ohe_feature.columns
        keep_features = feature_names[feature_names.isin(keep_values)]

        return ohe_feature[keep_features]
    

class DataCleaning(object):
    
    def stem_words(self, text):
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        return text

    def make_lower_case(self, text):
        return text.lower()

    def remove_stop_words(self, text):
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
        return text

    def remove_punctuation(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        text = tokenizer.tokenize(text)
        text = " ".join(text)
        return text

    def remove_emails(self, text):
        string_no_emails = re.sub("\S*@\S*\s?", "", text)
        return (string_no_emails)

    def remove_numbers(self, text):
        string_no_numbers = re.sub("\d+", "", text)
        return (string_no_numbers)