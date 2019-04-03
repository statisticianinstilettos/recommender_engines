import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender(object):

    def get_similar_recommendations(self, seed_item, feature_matrix, similarity_metric, n):
        ''' Return top n similar items to a seed item '''

        if similarity_metric not in ["cosine", "euclidean", "manhattan", "jaccard"]:
            return ValueError("similarity_metric must be cosine, euclidean, manhattan, or jaccard")

        # get style variant's feature vector
        item_vector = np.array(feature_matrix.loc[seed_item]).reshape(1, -1)

        # calculate similarity
        similarities = self._choose_similarity(item_vector, feature_matrix, similarity_metric)
        similarities = pd.DataFrame(similarities, index=feature_matrix.index.tolist())
        similarities.columns = ['similarity_score']
        similarities.sort_values('similarity_score', ascending=False, inplace=True)

        # return top n similar items with similarity scores
        similar_items = similarities.head(n).index.values.tolist()
        scores = similarities.head(n).similarity_score.values.tolist()

        return {"similar_items": similar_items, "score": np.round(scores, 5)}

    @staticmethod
    def _choose_similarity(item_vector, feature_matrix, similarity_metric):

        if similarity_metric == "cosine":
            similarities = 1 - pairwise_distances(X=feature_matrix, Y=item_vector, metric="cosine")
        elif similarity_metric == "euclidean":
            similarities = 1 - pairwise_distances(X=feature_matrix, Y=item_vector, metric="euclidean")
        elif similarity_metric == "manhattan":
            similarities = 1 - pairwise_distances(X=feature_matrix, Y=item_vector, metric="manhattan")
        elif similarity_metric == "jaccard":
            similarities = 1 - pairwise_distances(X=feature_matrix, Y=item_vector, metric="hamming")
        return similarities