
from sklearn.grid_search import GridSearchCV
import numpy as np
from scipy import io
from sklearn.linear_model import LogisticRegression

def model_selection(model, X, y, param_grid, **kwargs):
    clf = GridSearchCV(model, param_grid, **kwargs)
    clf.fit(X, y)
    return clf.best_estimator_
