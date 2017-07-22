# 3rd_Place_Solution_Kaggle_Mercedes
https://www.kaggle.com/c/mercedes-benz-greener-manufacturing

1. Run Prepare Data.py 
We use one-hot-encoding with thresholds set up for feature counts. Our dataset is relatively small given the high number 
of features and variability in Y-variable during CV testing. Therefore we want to avoid overfitting.
2. Run xgb_helper.py to import relevant XGB data
3. Run Ensemble Model.py
We use 6 different 1st Layer models and Ridge Regression as 2nd Layer model. We solve for Y after transforming it
to log(Y).
