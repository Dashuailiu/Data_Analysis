from pylab import *
from utilities import z_score
from Wine_quality import file_open, target_col
from utilities import FeatureSelector
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import metrics

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    macc = []

    for train_index, test_index in kf.split(data):
        # Filter training data
        train_predictors = data[predictors].iloc[train_index, :]
        # The target we’re using to train the algorithm
        train_target = data[outcome].iloc[train_index]
        # Training the algorithm using the predictors and target
        model.fit(train_predictors, train_target)
        # Record error from each cross−validation run
        macc.append(model.score(data[predictors].iloc[test_index, :], data[outcome].iloc[test_index]))
    print("Cross−Validation Score: %s" % "{0:.3%}".format(np.mean(macc)))
    return model, np.mean(macc)


def feature_selection(data):
    # Feature selection by Feature Important Light Boosting Model
    train = data.drop(columns=[target_col])
    fs = FeatureSelector(data=train, labels=data[target_col])
    fs.identify_zero_importance(task='classification', eval_metric='auc',
                                n_iterations=10, early_stopping=True)
    fs.plot_feature_importances(threshold=0.9, plot_n=12)
    plt.xlabel('Importance Value')
    plt.ylabel('Attributes')
    print(fs.feature_importances)
    ipt = 0
    sel_cols = []
    for i in range(len(fs.feature_importances.index)):
        if ipt >= 0.9:
            break
        ipt += fs.feature_importances['normalized_importance'].iloc[i]
        sel_cols.append(fs.feature_importances['feature'].iloc[i])
    print(list(enumerate(sel_cols)))
    return pd.concat([data[sel_cols], data[[target_col]]], axis=1)


def model_rforest(data):
    final_model = None
    # Normalization
    std_data = z_score(data, [target_col])
    # std_data = data
    # Modelling

    print("RandomForest Modeling")
    macc = []
    acc_temp = 0
    prefect_estimator = 10
    ne_list = range(50, 500, 10)
    for ne in ne_list:
        print("Optimization for parameter n_estimators %s" % ne)
        model = RandomForestClassifier(n_estimators=ne, max_features="auto",
                                       random_state=50, oob_score=True, n_jobs=-1)
        model, acc = classification_model(model, std_data, std_data.columns.drop(target_col), target_col)
        if acc > acc_temp:
            acc_temp = acc
            prefect_estimator = ne
        macc.append(acc)

    sns.lineplot(ne_list, macc)
    plt.axis("tight")
    plt.xlabel('Number of subtrees')
    plt.ylabel('Accuracy')
    plt.show()

    macc = []
    acc_temp = 0
    prefect_msl = 0
    msl_list = [i for i in range(1, 11, 1)] + [i for i in range(20, 80, 10)]
    for msl in msl_list:
        print("Optimization for parameter min_samples_leaf %s" % msl)
        model = RandomForestClassifier(n_estimators=prefect_estimator, min_samples_leaf=msl, max_features="auto",
                                       random_state=50, oob_score=True, n_jobs=-1)
        model, acc = classification_model(model, std_data, std_data.columns.drop(target_col), target_col)
        if acc > acc_temp:
            acc_temp = acc
            final_model = model
            prefect_msl = msl
        macc.append(acc)

    sns.lineplot(msl_list, macc)
    plt.axis("tight")
    plt.xlabel('Number of nodes of a subtree')
    plt.ylabel('Accuracy')
    plt.show()

    print("Final Model Accuracy %s" % "{0:.3%}".format(acc_temp))
    print(prefect_estimator)
    print(prefect_msl)
    return final_model


def modelling(data):
    # Random Forest
    data_sel = feature_selection(data)
    model_rf = model_rforest(data_sel)

    # Decision Tree


if __name__ == '__main__':
    data = file_open("./data/clean_data.csv")
    modelling(data)
