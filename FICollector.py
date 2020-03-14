import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize


class FICollectorMethod(object):
    def __init__(
            self,
            function,
            importance_handler,
            rank_handler,
            recalculate_rank=False,
            filter=None,
            remove_nan=False
    ):
        if filter is None:
            filter = {}
        self.filter = filter
        self.recalculate_rank = recalculate_rank
        self.remove_nan = remove_nan
        if callable(importance_handler) and callable(rank_handler) and callable(function):
            self.importance_handler = importance_handler
            self.rank_handler = rank_handler
            self.function = function
        else:
            raise ValueError

    def run(self, X, y):
        if self.remove_nan:
            return self.process_dropna(X, y)
        else:
            return self.process_all(X, y)

    def process_all(self, X, y):
        score = self.function(X, y)
        importance_row = self.importance_handler(score)
        if self.recalculate_rank:
            rank_row = self.rank_handler(score)
        else:
            rank_row = self.rank_handler(importance_row)
        return importance_row, rank_row

    def process_dropna(self, X, y):
        importance_row = np.array([])
        for column in X.T:
            df = pd.DataFrame({'X': column, 'y': y})
            df = df.dropna(axis=0)
            X_cleared = df["X"].values.reshape(-1, 1)
            y_cleared = df["y"].values
            score = self.function(X_cleared, y_cleared)
            importance = self.importance_handler(score)
            importance_row = np.concatenate((importance_row, importance), axis=0)
        rank_row = self.rank_handler(importance_row)
        return importance_row, rank_row


class FICollector(object):
    def __init__(self, features):
        self.features = features
        self.importance_df = pd.DataFrame(columns=features)
        self.rank_df = pd.DataFrame(columns=features)
        self.n_importance_df = pd.DataFrame(columns=features)
        self.feature_importance_methods = {}
        self.handlers = {}
        self.init_handlers()

    def init_handlers(self):
        def nan_handler(result):
            return np.full_like(self.features, np.NaN)

        def importance_base_handler(result):
            return result

        def importance_inline_handler(result, level=0):
            return result[level]

        def rank_base_handler(result):
            if type(result) is dict:
                result = np.array(list(result.values()))
            return np.argsort(np.argsort(result)).astype(int)

        def rank_inline_handler(result, level=0):
            return result[level]

        self.handlers['nan_handler'] = nan_handler

        self.handlers['importance_base_handler'] = importance_base_handler
        self.handlers['importance_inline_handler'] = importance_inline_handler

        self.handlers['rank_base_handler'] = rank_base_handler
        self.handlers['rank_inline_handler'] = rank_inline_handler

    def add_feature_importance_method(
            self,
            name,
            function,
            recalculate_rank=False,
            importance_handler="importance_base_handler",
            rank_handler="rank_base_handler",
            filter=None,
            remove_nan=False
    ):
        if not callable(function):
            raise ValueError
        if not callable(importance_handler):
            if importance_handler in self.handlers:
                importance_handler = self.handlers[importance_handler]
        if not callable(rank_handler):
            if rank_handler in self.handlers:
                rank_handler = self.handlers[rank_handler]
        self.feature_importance_methods[name] = FICollectorMethod(
            function,
            importance_handler,
            rank_handler,
            recalculate_rank=recalculate_rank,
            filter=filter,
            remove_nan=remove_nan
        )

    def run(self, X, y, methods=None, filter=None):
        if methods is not None:
            methods_to_run = {key: value for key, value in self.feature_importance_methods.items() if key in methods}
        elif filter is not None:
            methods_to_run = {}
            for method_key, method_value in self.feature_importance_methods.items():
                use_method = True
                for filter_key, filter_value in filter.items():
                    if filter_key not in method_value.filter or method_value.filter[filter_key] != filter_value:
                        use_method = False
                        break
                if use_method:
                    methods_to_run[method_key] = method_value
        else:
            methods_to_run = self.feature_importance_methods
        for name, method in enumerate(methods_to_run):
            importance_tuple = self.feature_importance_methods[method].run(X, y)
            for index, row in enumerate(importance_tuple):
                if not type(row) is dict:
                    dict_row = {}
                    for i, importance in enumerate(row):
                        dict_row[self.features[i]] = importance
                    row = dict_row
                row = pd.Series(row, name=method)
                if index == 0:
                    self.importance_df = self.importance_df.append(row)
                else:
                    self.rank_df = self.rank_df.append(row)
        self.normalize_importance_dataframe()

    def get_sorted_features(
            self,
            limit=10,
            sort_by="normalized_importance",
            features_only=True,
            ascending=False
    ):
        if sort_by == "normalized_importance":
            sum = self.n_importance_df.sum(axis=0, skipna=True)
        elif sort_by == "rank":
            sum = self.rank_df.sum(axis=0, skipna=True)
        else:
            raise ValueError("sort_by should be \"normalized_importance\" or \"rank\"")
        sum = sum.sort_values(ascending=ascending)
        if limit > 0:
            sum = sum.iloc[:limit]
        sum = sum.to_dict()

        return list(sum.keys()) if features_only else sum

    def normalize_importance_dataframe(self):
        for index, row in self.importance_df.iterrows():
            row = row.fillna(0)
            n = normalize(row.values.reshape(1, -1))
            series = pd.Series(n[0], name=index, index=self.n_importance_df.columns)
            self.n_importance_df = self.n_importance_df.append(series)
