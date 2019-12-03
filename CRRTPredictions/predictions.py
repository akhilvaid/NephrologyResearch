#!/bin/python

import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split


class Criteria:
    outcome_var = 'OUTCOME_DEATH'


class Models:
    xgb = XGBClassifier(gpu_id=0, n_jobs=-1)


# Reshape time series data into the usual kind of dataframe for predictions
def reshape(df_time_series, df_constant_features, scale=True, head=None):
    try:
        # Drop the constant features from the times series data
        constant_features = df_constant_features.columns
        df_time_series = df_time_series.drop(constant_features, axis=1)
        df_time_series = df_time_series.drop('TIMESTEPS', axis=1)
    except KeyError:
        pass

    # Consider these many timesteps
    if head:
        df_time_series = df_time_series.groupby('HADM_ID').head(head)

    # Make the rows > columns conversion
    # Groupby and then transpose for those groups
    def index_group(df):
        return df.reset_index(drop=True).T

    df_time_series_reshaped = df_time_series.groupby('HADM_ID').\
        apply(index_group).unstack()
    df_time_series_reshaped = df_time_series_reshaped.replace(np.nan, 0)
    df_time_series_reshaped.columns = [
        i for i in range(df_time_series_reshaped.shape[1])]

    # Merge with constant features again
    df_time_series_reshaped = df_time_series_reshaped.merge(
        df_constant_features, left_index=True, right_index=True, how='left')

    if scale:
        scaled = MinMaxScaler().fit_transform(df_time_series_reshaped)
        df_time_series_reshaped = pd.DataFrame(
            scaled,
            index=df_time_series_reshaped.index,
            columns=df_time_series_reshaped.columns)

    return df_time_series_reshaped


def attach_outcomes(df_features, df_outcomes):
    df_outcomes = df_outcomes.set_index('HADM_ID')[[Criteria.outcome_var]]
    df_features = df_features.merge(
        df_outcomes, left_index=True, right_index=True, how='left')

    return df_features


def predict(df):
    print('Cross validating...')
    X = df.drop(Criteria.outcome_var, axis=1)
    y = df[Criteria.outcome_var].values.ravel()

    cvs = cross_val_score(Models.xgb, X, y, cv=5)
    print(cvs.mean(), cvs.std())


def roc(df):
    print('Calculating ROC...')
    train, test = train_test_split(df)

    train_x = train.drop(Criteria.outcome_var, axis=1)
    train_y = train[[Criteria.outcome_var]].values.ravel()
    test_x = test.drop(Criteria.outcome_var, axis=1)
    test_y = test[[Criteria.outcome_var]].values.ravel()

    Models.xgb.fit(train_x, train_y)
    pred = Models.xgb.predict_proba(test_x)

    fpr = dict()
    tpr = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(test_y, pred[:, 1])

    plt.figure()
    plt.plot(fpr[1], tpr[1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()

    _roc = roc_auc_score(test_y, pred[:, 1])
    print(_roc)


def postprocess_predictions(
        df_timeseries, df_constant_features,
        df_outcomes, dict_predictions,
        start_interval):

    print()
    df_timeseries = df_timeseries.drop('TIMESTEPS', axis=1)

    for this_hadm in dict_predictions:
        df_predictions = dict_predictions[this_hadm]
        df_predictions['HADM_ID'] = this_hadm

        df_ts_train = df_timeseries.reset_index().query('HADM_ID != @this_hadm')
        df_ts_predict = df_timeseries.reset_index().query('HADM_ID == @this_hadm')

        was_outcome_positive = df_outcomes.query(
            'HADM_ID == @this_hadm')[Criteria.outcome_var].iloc[0]
        print(this_hadm, 'Recorded outcome:', was_outcome_positive)

        # Current prediction
        df_train_now = reshape(df_ts_train, df_constant_features, True, start_interval)
        df_train_now = attach_outcomes(df_train_now, df_outcomes)
        df_ground_truth_now = reshape(
            df_ts_predict, df_constant_features, True, start_interval)

        clf = XGBClassifier(gpu_id=0, n_jobs=-1)
        X = df_train_now.drop(Criteria.outcome_var, axis=1)
        y = df_train_now[[Criteria.outcome_var]].values.ravel()
        clf.fit(X, y)

        all_probabilities = []

        gt_probability = clf.predict_proba(df_ground_truth_now)[:, 1][0]
        all_probabilities.append(gt_probability)
        print('Probability now:', gt_probability)

        # Future predictions
        for i in range(df_predictions.shape[0]):
            df_train_at_time = reshape(
                df_ts_train, df_constant_features, True, start_interval + i + 1)
            df_train_at_time = attach_outcomes(df_train_at_time, df_outcomes)

            # Attach new predictions
            df_ground_truth_at_time = reshape(
                df_ts_predict, df_constant_features, True, start_interval + i + 1)

            df_predict_at_time = df_ts_predict.head(start_interval)
            df_predict_at_time = df_predict_at_time.append(df_predictions.iloc[:(i + 1)], sort=True)
            df_predict_at_time = reshape(df_predict_at_time, df_constant_features, True, False)

            # Train and predict
            clf = XGBClassifier(gpu_id=0, n_jobs=-1)
            X = df_train_at_time.drop(Criteria.outcome_var, axis=1)
            y = df_train_at_time[[Criteria.outcome_var]].values.ravel()
            clf.fit(X, y)

            pred_probability = clf.predict_proba(df_predict_at_time)[:, 1][0]
            # gt_probability = clf.predict_proba(df_ground_truth_at_time)[:, 1][0]

            # print('Probability at interval', i + 1, pred_probability, '/', gt_probability)
            all_probabilities.append(pred_probability)

        print(all_probabilities)
