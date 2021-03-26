import pickle
import time
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, \
    RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def load_data(filepath, time_range=[], coin='BTC'):
    data = pickle.load(open(filepath, 'rb'))
    data.rename({'text_polarity_len': 'len'}, axis='columns', inplace=True)
    if coin != '':
        print('Filter by cryptocurrency coin:', coin)
        data = data[data.coin == coin]
    if len(time_range) > 0:
        print('Filter by time range:', time_range)
        data = data.loc[(data.index >= time_range[0]) & (data.index <= time_range[1])]
    data = data.sort_index()
    len_data = len(data)
    print(data.tail())
    print(data.columns[data.isna().any()].tolist())
    data = data.dropna()
    print('Dropped NaN:', len_data, '->', len(data))
    return data


def pre_process(data, text_features, max_lag=7):
    if 'len' in data.columns:
        for col in data.columns:
            data.loc[((data.index < '2017-07-11') | (data.index > '2017-07-23')) & data[col].isna(), 'len'] = 0
    data = data.replace(np.inf, 0)

    Y = data['direction']
    Y = Y.astype(bool)

    data = data.interpolate()

    ma_cols = ['interest'] + text_features
    ma = data[data.columns.intersection(ma_cols)].rolling(14).mean()
    non_senti = data[data.columns.difference(ma_cols)]
    data = pd.concat([non_senti, ma], axis=1)

    non_diff = ['return', 'volumefrom'] + [t for t in text_features if 'tot_posneg' in t]
    diff = ['volumeto', 'spread', 'interest'] + text_features
    data_nd = data[data.columns.intersection(non_diff)]
    data_d = data[data.columns.intersection(diff)].diff()
    data = pd.concat([data_nd, data_d], axis=1, sort=False)
    data = data.dropna()

    # Create lagging variables
    lags = max_lag
    print("Lagging variables, max_lag:", max_lag)
    X = pd.DataFrame()
    for i in range(lags):
        lag = data.shift(i + 1).add_suffix('_' + str(i + 1))
        X = pd.concat([X, lag], axis=1)
    X = X.dropna()
    y = Y.loc[Y.index >= X.index[0]]
    return X, y


def train_model(X, y, model, param_grid, model_type, train_test_prop=0.8, n_splits=10, standardize=True, score_metric='roc_auc',
                tuning='rand'):
    # Split into train and test using nested time-series cross-validation
    cv_split = int(len(X) * train_test_prop)
    X_train, X_test = X.iloc[:cv_split], X.iloc[cv_split:]
    y_train, y_test = y.iloc[:cv_split], y.iloc[cv_split:]
    print('Train size:', len(X_train), len(y_train), 'test size:', len(X_test), len(y_test))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if standardize:
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('clf', model)
        ])
    else:
        pipe = Pipeline([
            ('clf', model)
        ])

    if tuning == 'rand':
        grid_model = RandomizedSearchCV(estimator=pipe,
                                        param_distributions=param_grid,
                                        n_iter=200,
                                        cv=tscv,
                                        refit=score_metric,
                                        scoring=['accuracy', 'roc_auc', 'f1_weighted'],
                                        return_train_score=True)
    else:
        grid_model = GridSearchCV(estimator=pipe,
                                  param_grid=param_grid,
                                  cv=tscv,
                                  scoring=['accuracy', 'roc_auc', 'f1_weighted'],
                                  refit=score_metric,
                                  return_train_score=True)

    print('Started training.')
    start_time = time.time()
    grid_model.fit(X_train, y_train)
    results = grid_model.cv_results_
    elapsed_time = time.time() - start_time
    print('Finished training. Time:', elapsed_time)

    clf = grid_model.best_estimator_
    validation_accuracy = np.mean(results['mean_test_accuracy'])
    validation_roc_auc = np.mean(results['mean_test_roc_auc'])
    validation_f1_weighted = np.mean(results['mean_test_f1_weighted'])
    print('Prediction performance on validation set')
    print('\tAccuracy:', np.round(validation_accuracy, 3))
    print('\tROC AUC score:', np.round(validation_roc_auc, 3))
    print('\tF1 (weighted) score:', np.round(validation_f1_weighted, 3))
    if model_type == 'rf' or model_type == 'xgboost':
        md = SelectFromModel(clf.named_steps['clf'], prefit=True)  # , threshold = thresh)
        names = X.iloc[:, md.get_support()].columns.values
        feat_imp = (clf.named_steps['clf'].feature_importances_, names)
    elif model_type == 'logreg':
        feat_imp = clf.named_steps['clf'].coef_
    elif model_type == 'svm':
        md = SelectFromModel(clf.named_steps['clf'], prefit=True)
        if md.kernel == 'linear':
            names = X.iloc[:, md.get_support()].columns.values
            feat_imp = (clf.named_steps['clf'].coef_, names)
        else:
            feat_imp = None
    else:
        feat_imp = None
    print('Feature importance', feat_imp)
    pred_results = {'Model': clf,
                    'Feature importances': feat_imp,
                    'Parameter value': grid_model.best_params_,
                    'Prediction accuracy': validation_accuracy,
                    'AUC score': validation_roc_auc,
                    'F1 score': validation_f1_weighted}
    return pred_results


def get_model(data, parameters):
    text_features_polarity = ['text_polarity_negative',
                              'text_polarity_positive', 'text_polarity_sum',
                              'text_polarity_tot_neg', 'text_polarity_tot_pos',
                              'text_polarity_tot_posneg', 'text_polarity_posneg',
                              'sentiment_scores_0_mean', 'sentiment_scores_1_mean']
    text_features_subjectivity = ['text_subjectivity_negative', 'text_subjectivity_positive',
                                  'text_subjectivity_sum', 'text_subjectivity_tot_neg', 'text_subjectivity_tot_posneg',
                                  'text_subjectivity_posneg', 'text_subjectivity_tot_pos']
    text_features_topic = ['topic' + str(i) + '_mean' for i in range(0, 15)] + ['topic_scores_' + str(i) + '_mean' for i
                                                                                in range(0, 15)]


    other_text_features = ['len']
    all_text_features = other_text_features + text_features_topic + text_features_polarity + text_features_subjectivity
    selected_text_features = [other_text_features]
    if parameters['text_features'] is not None:
        if parameters['text_features'] == 'all':
            selected_text_features = text_features_polarity + text_features_subjectivity + other_text_features + text_features_topic
        elif parameters['text_features'] == 'none':
            selected_text_features = []
        else:
            if 'polarity' in parameters['text_features']:
                selected_text_features = selected_text_features + text_features_polarity
            if 'subjectivity' in parameters['text_features']:
                selected_text_features = selected_text_features + text_features_subjectivity
            if 'topic' in parameters['text_features']:
                selected_text_features = selected_text_features + text_features_topic
    used_text_features = list(set(selected_text_features).intersection(set(data.columns)))
    if len(selected_text_features) > len(used_text_features):
        print(
            'WARNING: not all text features used, because the corresponding columns are not in the basetable dataframe. 2 main reasons: wrong column\'s names or these features were not extracted for this dataset.')
        print('Omitted columns:', list(set(selected_text_features) - set(used_text_features)))

    data = data.drop(list(set(all_text_features).intersection(set(data.columns)) - set(used_text_features)), axis=1)
    print('Features:', data.columns)
    X, y = pre_process(data, used_text_features, parameters['max_lag'])
    print('Dataset size:', len(X), 'from', min(data.index), 'to', max(data.index), '| Target class balance:',
          np.round(sum(y) / len(y), 3))
    n_alphas = 1000

    random_state = 456

    if parameters['model_type'] is not None:
        if parameters['model_type'] == 'svm':
            n_alphas = 100
            clf = SVC(random_state=random_state, kernel='rbf')
            param_grid = {'clf__C': np.logspace(-2, 10, n_alphas), 'clf__gamma': np.logspace(-9, 3, n_alphas)}
        if parameters['model_type'] == 'nb':
            clf = GaussianNB(random_state=random_state)
        if parameters['model_type'] == 'logreg':
            clf = LogisticRegression(random_state=random_state, solver='liblinear')
            param_grid = {'clf__C': np.logspace(-10, 10, n_alphas)}
        if parameters['model_type'] == 'rf':
            clf = RandomForestClassifier(random_state=random_state, n_estimators = 50)
            if parameters['tuning_mode'] == 'rand':
                param_grid = {"clf__max_depth": sp_randint(1, 25),
                              "clf__min_samples_leaf": sp_randint(2, 50),
                              "clf__min_samples_split": sp_randint(2, 56),
                              "clf__max_features": sp_randint(1, X.shape[1])}
            else:
                param_grid = {"clf__max_depth": sp_randint(1, 25),
                              "clf__min_samples_leaf": sp_randint(2, 50),
                              "clf__min_samples_split": sp_randint(2, 56),
                              "clf__max_features": sp_randint(1, X.shape[1])}
        if parameters['model_type'] == 'xgboost':
            clf = GradientBoostingClassifier(random_state=random_state, n_estimators = 50)
            param_grid = {
                "clf__learning_rate": [0.01, 0.075, 0.1, 0.2],
                "clf__min_samples_split": np.linspace(0.1, 1, 10),
                "clf__min_samples_leaf": np.linspace(0.1, 0.5, 5),
                "clf__max_depth": np.linspace(1, 32, 32, endpoint=True),
                "clf__max_features": ["log2", "sqrt"],
                "clf__subsample": [0.5, 0.85, 0.9]
            }
    else:
        n_alphas = 100
        clf = SVC(random_state=random_state, kernel='rbf')
        param_grid = {'clf__C': np.logspace(-2, 10, n_alphas), 'clf__gamma': np.logspace(-9, 3, n_alphas)}

    pred_results = train_model(X, y, clf, train_test_prop=parameters['train_test_prop'], param_grid=param_grid,
                               tuning=parameters['tuning_mode'], score_metric=parameters['score_metric'],
                               standardize=parameters['standardize'], model_type=parameters['model_type'])
    if parameters['save']:
        filename = '_'.join(
            [parameters['model_name'], parameters['model_type'], parameters['text_features'], parameters['tuning_mode'],
             parameters['score_metric'], parameters['coin_name']])
        pickle.dump(pred_results, open('../models/' + filename + '.p', 'wb'))
    return pred_results


# python TrainModel.py -f ../../data/processed/bitcointalk/date_no_structure/new_basetable_nonan.p -t rand -tf none
# python TrainModel.py -f ../../data/processed/news/JST_new_basetable.p -m logreg -t rand -tf none
# python TrainModel.py -f ../../data/processed/bitcointalk/date_no_structure/JST_new_basetable.p -m logreg -t rand -tf all -p 0.8
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--filename", dest="basetable_filename",
                        help="Basetable filename", default="new_basetable")
    parser.add_argument("-m", "--model", dest="model_type",
                        help="Machine learning model type (svm / nb / logreg / rf / xgboost)", default='xgboost')
    parser.add_argument("-t", "--tuning", dest="tuning_mode",
                        help="Cross-validation optimization mode (grid / rand)", default='rand')
    parser.add_argument("-st", "--standardize", dest="standardize", type=bool,
                        help="Standardize or not", default=True)
    parser.add_argument("-p", "--prop", dest="train_test_prop", type=float,
                        help="Train/test dataset size proportion (default: 0.8)", default=0.8)
    parser.add_argument("-tf", "--text_features", dest="text_features",
                        help="Which text features to use (none / all / polarity / topic / subjectivity). Can combine via + (e.g., polarity+topic). if not specified, only length of texts is used.")
    parser.add_argument("-tr", "--timerange", dest="time_range",
                        help="Time range string in the following format: 31/12/2019-31/12/2020 (can also say 'default' to work with 01/01/2017-30/11/2017")
    parser.add_argument("-s", "--score", dest="score_metric",
                        help="Score metric to guide grid search (roc_auc / accuracy / f1)", default='roc_auc')
    parser.add_argument("-sv", "--save", dest="save", type=bool,
                        help="Save the model file", default=True)
    parser.add_argument("-a", "--all", dest="all", type=bool,
                        help="Run all combinations of models and text features.", default=False)
    parser.add_argument("-n", "--model_name", dest="model_name",
                        help="Model name for saving the files.", default="crypto-no-sacred")
    parser.add_argument("-c", "-coin_name", dest="coin_name",
                        help="Cryptocurrency coin to predict for.", default="")
    parser.add_argument("-ml", "-max_lag", dest="max_lag", default=7, type=int)

    args = parser.parse_args()
    args = vars(args)
    print('Basetable:', args['basetable_filename'])
    print('Model:', args['model_type'], '| Train/test:', args['train_test_prop'], '| Text features:',
          args['text_features'], '| Tuning mode:', args['tuning_mode'])
    if args['time_range'] is not None:
        if args['time_range'] == 'default':
            time_range = [pd.Timestamp.strptime("01/01/2017", "%d/%m/%Y"),
                          pd.Timestamp.strptime("30/11/2017", "%d/%m/%Y")]
        else:
            start_time, end_time = args['time_range'].split('-')
            time_range = [pd.Timestamp.strptime(start_time, "%d/%m/%Y"), pd.Timestamp.strptime(end_time, "%d/%m/%Y")]
    else:
        time_range = []
    data = load_data(args['basetable_filename'], time_range, args['coin_name'])
    if args['all']:
        for model_type in ['nb', 'logreg', 'rf', 'svm']:
            for text_features in ['none', 'all', 'polarity', 'topic', 'subjectivity']:
                parameters = args
                parameters['model_type'] = model_type
                parameters['text_features'] = text_features
                print(parameters)
                get_model(data, parameters)
    else:
        print(args)
        get_model(data, args)
