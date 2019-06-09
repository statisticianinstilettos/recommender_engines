import pandas as pd
import numpy as np
from fastFM import sgd
from fastFM import als
import scipy.sparse as sps


class FactorizationMachinesPrediction(object):

    def __init__(self, items, feature_names, fm):
        '''
        Args:
            items: list of all unique intems the FM is trained on.
            feature_names: feature names from the trained model.
            fm: trained Factorization Machine Classification model
        '''
        self.items = items
        self.feature_names = feature_names
        self.X = pd.DataFrame([], columns=self.feature_names, index=range(len(self.items)))
        self.fm = fm

    def predict_classification(self, userid, observed_features, n):
        ''' Return top n recommended items for a user
        Args:
            userid: id of user we are requesting predictions for.
            observed_features: dictionary of observed feature names and their values.
            n: number of similar items to return.
        Returns: predictions in json format
        '''

        X_observation = self._format_X(userid, observed_features)

        X_sparse = sps.csc_matrix(X_observation)

        pred_proba = self.fm.predict_proba(X_sparse)
        pred_proba = np.round(pred_proba, 5)

        #format predictions into ranked list
        pred = pd.DataFrame(self.ordered_item_list)
        pred.columns = ["items"]
        pred["score"] = pred_proba
        pred.sort_values("score", inplace=True, ascending=False)
        pred = pred.head(n)

        return {"items": pred['items'].tolist(), "score": pred['score'].tolist()}

    def _format_X(self, userid, observed_features):

        X_observation = self.X.copy()

        #ohe items
        for i in range(len(self.items)):
            X_observation.loc[i, self.items[i]] = 1

        #get ordered list of items
        x = X_observation.stack()
        self.ordered_item_list = pd.Series(pd.Categorical(x[x != 0].index.get_level_values(1))).tolist()

        #ohe user
        if userid in self.feature_names:
            X_observation.loc[:, userid] = 1

        #fill rest of sparse matrix with feature values
        for k,v in observed_features:
            if k in self.feature_names:
                X_observation.loc[:, k] = v

        X_observation.fillna(0, inplace=True)

        return X_observation



class TrainFactorizationMachines(object):

    @staticmethod
    def fit_classification(X_train, y_train, method, random_state=0, n_iter=100, init_stdev=0.1, l2_reg_w=0, l2_reg_V=0, rank=2, step_size=None):
        '''Uses als or sgd to train a binary classification model using FM.
        Args:
            X_train: pandas dataframe.
            y_train: pandas series.
            method: 'sgd' or 'als'.
        Returns: fm, feature_names
        '''

        if method not in ['sgd', 'als']: raise ValueError('method must be "sgd" or "als".')

        if method == 'sgd' and step_size == None: raise ValueError('step size must be defined for "sgd".')

        feature_names = X_train.columns.tolist()
        X_train_sparse = sps.csc_matrix(X_train)

        if method == "sgd":
            fm = sgd.FMClassification(n_iter=n_iter,
                                      init_stdev=init_stdev,
                                      l2_reg_w=l2_reg_w,
                                      l2_reg_V=l2_reg_V,
                                      rank=rank,
                                      step_size=step_size,
                                      random_state=random_state)
            fm.fit(X_train_sparse, y_train)

        if method == "als":
            fm = als.FMClassification(n_iter=n_iter,
                                  init_stdev=init_stdev,
                                  rank=rank,
                                  l2_reg_w=l2_reg_w,
                                  l2_reg_V=l2_reg_V,
                                  random_state=random_state)
            fm.fit(X_train_sparse, y_train)

        return fm, feature_names

