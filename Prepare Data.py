import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import r2_score
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
id_test = test.ID
y_train = train['y'].values

num_train=train.shape[0]
train.drop(['y'],inplace=True,axis=1)


# Load all categorical data using one hot encoding
da=pd.concat([train,test])
threshold = 10 # Anything that occurs less than this will be removed.

for c in da.columns:
    if da[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(da[c].values))
        da[c] = lbl.transform(list(da[c].values))
        value_counts = da[c].value_counts() # Specific column 
        to_remove = value_counts[value_counts <= threshold].index
        if len(to_remove) > 0:
            da[c].replace(to_remove, 0, inplace=True)
            print(to_remove, c)

to_remove1=[]
to_remove2=[]

# Remove all features appearing less than 20 times to avoid overfitting
sums=da.sum(axis=0)
to_remove1=sums[sums<20].index.values
da=da.loc[:,da.columns.difference(to_remove1)]


train = da[:num_train]
test = da[num_train:]

X_train = train.values
X_test = test.values

n_comp = 10

# -------- Create extra features --------
# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp)
pca2_results_train = pca.fit_transform(train)
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train)
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

#save columns list before adding the decomposition components

usable_columns = list(set(train.columns))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

#usable_columns = list(set(train.columns) - set(['y']))


#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values

X_train = finaltrainset
X_test = finaltestset

