import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import plot_importance
from matplotlib import pyplot
import operator
from sklearn.model_selection import GridSearchCV
import random
import string
from scipy.stats import boxcox
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math
from sklearn.metrics import log_loss
from collections import defaultdict
from sklearn.metrics import r2_score






def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000, stopping = 80):
    print("will stop after", stopping)
    num_rounds = num_rounds

    plst = list(xgb_params.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    
    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=stopping)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model
    





