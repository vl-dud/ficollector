# FICollector

FICollector is a tool for feature selection in Python.

## Requirements
* pandas
* numpy
* sklearn

## How to use

```python
from sklearn import datasets

# load a testing dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# initialize your collector
feature_importance_collector = FICollector(X.columns)

# add methods for feature importance calculation
feature_importance_collector.add_feature_importance_method(
    'mutual_info_classif',
    fs.mutual_info_classif,
)

feature_importance_collector.add_feature_importance_method(
    'f_classif',
    fs.f_classif,
    importance_handler="importance_inline_handler",
)

# run feature importances calculation
feature_importance_collector.run(
    X.to_numpy(),
    y.to_numpy(),
)

# show feature importances dataframe 
print(feature_importance_collector.importance_df)
# show normalized feature importances dataframe 
print(feature_importance_collector.n_importance_df)
# show dataframe with features ranks
print(feature_importance_collector.rank_df)

# get three best features
best_3_features = feature_importance_collector.get_sorted_features(limit=3)
```