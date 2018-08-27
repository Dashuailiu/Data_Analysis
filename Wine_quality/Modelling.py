from pylab import *
from utilities import z_score
from Wine_quality import file_open, target_col
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import metrics

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def classification_model(model, data, predictors, outcome):
    # Fit the model:
    model.fit(data[predictors], data[outcome])
    # Make predictions on training set
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy:%s" % "{0:.3%}".format(accuracy))
    # Perform k−fold cross−validation with 5 folds
    kf = KFold(n_splits=5)
    error = []

    for train_index, test_index in kf.split(data):
        # Filter training data
        train_predictors = (data[predictors].iloc[train_index, :])
        # The target we’re using to train the algorithm
        train_target = data[outcome].iloc[test_index]
        # Training the algorithm using the predictors and target
        model.fit(train_predictors, train_target)
        # Record error from each cross−validation run
        error.append(model.score(data[predictors].iloc[test_index, :], data[outcome].iloc[test_index]))
        print("Cross−Validation Score: % s" % "{0:.3%}".format(np.mean(error)))
    return model


def model_rforest(data):
    # Feature selection by Feature Important
    feature_model = ExtraTreesClassifier()
    features = data.iloc[:, :data.shape[1]-1]
    target = data[target_col]
    feature_model.fit(features, target)
    feature_o = feature_model.feature_importances_
    print(feature_o)
    feature_model = SelectFromModel(feature_model, prefit=True)
    data_new = feature_model.transform(features)
    data_new = pd.concat([data_new, target])
    print(data_new.head())

    # Normalization
    std_data = z_score(data_new, [target_col])
    # Modelling
    model = RandomForestClassifier()
    #classification_model(model, std_data, std_data.columns.drop(target_col), target_col)


def modelling(data):
    # Random Forest
    model_rforest(data)


if __name__ == '__main__':
    data = file_open("./data/clean_data.csv")
    modelling(data)
