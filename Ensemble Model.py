
# We set up ensemble CV using 6 models:
    # XGBoost
    # LGB
    # ExtraTrees
    # RandomForest
    # GBM
    # Lasso
    
# We run Ridge Regression as our second level model on the outputs 

 #---------------------------------------------------------------------
 
 
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostClassifier, RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesRegressor
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import random



# Train on logarithms of Y as it deals better with highly sparce data
y_train = np.log(y_train)



# -------- STEP 1: SET UP PARAMETERS FOR ENSEMBLE -----------
xgb_params = {
    'n_trees': 1000, 
    'eta': 0.01,
    'max_depth': 4,
    'subsample': 0.8,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': np.mean(y_train), # base prediction = mean(target)
    'silent': 1
}
num_rounds = 1200


# -------------------


RS=1
np.random.seed(RS)
ROUNDS = 1000 # 1300,1400 all works fine
params = {
    'objective': 'regression',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'learning_rate': 0.01 , #small learn rate, large number of iterations
        'verbose': 0,
        'num_leaves': 2**4,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.7,
        'feature_fraction_seed': 1,
        'max_bin': 70,
        'max_depth': 3,
        'num_rounds': ROUNDS,
    }

cv_scores = []
scores1 = []
scores2 = []
scores3 = []
scores4 = []
scores5 = []
scores6 = []

val_scores = []
X_pred = []
X_pred1 = []
X_pred2 = []
X_pred3 = []
X_pred4 = []
X_pred5 = []
X_pred6 = []

y_pred = []
X_train_shuffled = []


kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=999)

# -------- STEP 2: RUN MODELS IN CV AND SAVE RESULTS INTO ARRAYS -----------

fold_no = 0
for dev_index, val_index in kf.split(range(X_train.shape[0])):
    fold_no = fold_no + 1
    print("training fold number" ,fold_no)
    dev_X, val_X = X_train[dev_index,:], X_train[val_index,:]
    dev_y, val_y = y_train[dev_index], y_train[val_index]
    sup_train_part = sup_train[val_index,:]
    y_pred = np.append(y_pred, val_y)
        
    # get new dataset of shuffled variables
    if len(X_train_shuffled) < 1:
        X_train_shuffled = val_X
        sup_train_shuffled = np.array(sup_train_part)
    else:
        X_train_shuffled = np.row_stack((X_train_shuffled, val_X))
        sup_train_shuffled = np.row_stack((sup_train_shuffled, sup_train_part))

    # model 1
    preds, model = runXGB(dev_X, dev_y, val_X, num_rounds=num_rounds)
    if len(X_pred1) < 1:
        X_pred1 = preds
    else:
        X_pred1 = np.concatenate((X_pred1, preds), axis = 0)
    
    scores1.append(r2_score(np.exp(val_y), np.exp(preds)))
    print("model 1 scores: ", scores1)
    
    # model 2
    train_lgb = lgb.Dataset(dev_X,dev_y)
    model=lgb.train(params,train_lgb,num_boost_round=num_rounds)
    preds = model.predict(val_X)
    
    if len(X_pred2) < 1:
        X_pred2 = preds
    else:
        X_pred2 = np.concatenate((X_pred2, preds), axis = 0)

    scores2.append(r2_score(np.exp(val_y), np.exp(preds)))
    print("model 2 scores: " ,scores2)

    # models other
    
    clf1 = ExtraTreesRegressor(n_estimators=1000, max_depth=4, min_samples_leaf=1)
    clf2 = RandomForestRegressor(n_estimators=1000, max_depth = 7, min_samples_split= 20, random_state=0)
    clf3 = GradientBoostingRegressor(learning_rate= 0.003, max_depth = 3,min_samples_split= 35, min_samples_leaf = 10, n_estimators= 1500)
    clf4 = LassoLarsCV(cv=20)

    
    clf1.fit(dev_X, dev_y)
    preds = clf1.predict(val_X)
    if len(X_pred3) < 1:
        X_pred3 = preds
    else:
        X_pred3 = np.concatenate((X_pred3, preds), axis = 0)
    
    scores3.append(r2_score(np.exp(val_y), np.exp(preds)))
    print("model 3 scores: " ,scores3)

    clf2.fit(dev_X, dev_y)
    preds = clf2.predict(val_X)
    if len(X_pred4) < 1:
        X_pred4 = preds
    else:
        X_pred4 = np.concatenate((X_pred4, preds), axis = 0)
        
    scores4.append(r2_score(np.exp(val_y), np.exp(preds)))
    print("model 4 scores: " ,scores4)

    clf3.fit(dev_X, dev_y)
    preds = clf3.predict(val_X)
    if len(X_pred5) < 1:
        X_pred5 = preds
    else:
        X_pred5 = np.concatenate((X_pred5, preds), axis = 0)
        
    scores5.append(r2_score(np.exp(val_y), np.exp(preds)))
    print("model 5 scores: " ,scores5)

    clf4.fit(dev_X, dev_y)
    preds = clf4.predict(val_X)
    if len(X_pred6) < 1:
        X_pred6 = preds
    else:
        X_pred6 = np.concatenate((X_pred6, preds), axis = 0)
        
    scores6.append(r2_score(np.exp(val_y), np.exp(preds)))
    print("model 6 scores: " ,scores6)



print(np.average(scores1))
print(np.average(scores2))
print(np.average(scores3))
print(np.average(scores4))
print(np.average(scores5))
print(np.average(scores6))






# -------- STEP 3: RUN MODELS ON THE TEST DATA-----------

X_pred = np.column_stack((X_pred1, X_pred2, X_pred3,X_pred4,X_pred5, X_pred6))
X_pred_full = np.column_stack((X_pred, sup_train_shuffled))


X_pred_test = []

preds1, model = runXGB(X_train, y_train, X_test, num_rounds=num_rounds)

train_lgb = lgb.Dataset(X_train,y_train)
model=lgb.train(params,train_lgb,num_boost_round=num_rounds)
preds2 = model.predict(X_test)

clf1.fit(X_train, y_train)
preds3 = clf1.predict(X_test)

clf2.fit(X_train, y_train)
preds4 = clf2.predict(X_test)

clf3.fit(X_train, y_train)
preds5 = clf3.predict(X_test)

clf4.fit(X_train, y_train)
preds6 = clf4.predict(X_test)

X_pred_test = np.column_stack((preds1, preds2, preds3, preds4, preds5, preds6))
X_pred_test_full = np.column_stack((X_pred_test, sup_test))


preds1 = np.exp(preds1)
preds2 = np.exp(preds2)
preds3 = np.exp(preds3)
preds4 = np.exp(preds4)
preds5 = np.exp(preds5)
preds6 = np.exp(preds6)


# -------- STEP 4: RUN LEVEL 2 MODEL -----------


print("training ensembler")

from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 1, solver = 'lsqr')

# ---------- LOG Change ------------
y_pred = np.exp(y_pred)
ridge.fit(X_pred, y_pred)
preds = ridge.predict(X_pred_test)
preds = np.exp(preds)


sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = preds
sub.to_csv('Ensemble_solution.csv', index=False)






