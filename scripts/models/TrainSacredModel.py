import os
import pickle
import random as rn
import sys
import time
import warnings

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from imblearn.over_sampling import SMOTE
from keras import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Concatenate, LSTM, Input, Embedding, TimeDistributed, BatchNormalization, Activation, \
    AlphaDropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import array, hstack
from sacred import Experiment
from scipy.stats import randint as sp_randint
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, \
    RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils import class_weight
from tqdm import tqdm

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from helper_functions import determine_data_source


def warn(*args, **kwargs):
    pass


SEED = 42
MAX_LENGTH = 200
REPORT_FILENAME = 'D:/crypto_DL_bitcointalk_reg_classweights_seeded.log'
__author__ = "Ekaterina Loginova"
__email__ = "ekaterina.loginova@ugent.be"
__status__ = "Development"

ex = Experiment('crypto_model')
warnings.warn = warn


def calculate_disbalance(y, y_pred):
    return sum(np.round(y_pred) / len(y_pred))


def seed_everything(seed=42):
    rn.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.set_random_seed(SEED)
    except Exception as e:
        print('No tensorflow installed => not seeding.')


def load_embeddings(embedding_path='D:/embeddings/glove.twitter.27B/glove.twitter.27B.50d.txt'):
    glove_file = datapath(embedding_path)
    tmp_file = get_tmpfile("test_word2vec.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    word_vectors = KeyedVectors.load_word2vec_format(tmp_file)
    return word_vectors


def report_metrics(y_true, y_pred):
    y_pred_round = [int(np.round(x)) for x in y_pred]
    acc = accuracy_score(y_true=y_true, y_pred=y_pred_round)
    print('Accuracy: {:.3f}'.format(acc))
    f1 = f1_score(y_true=y_true, y_pred=y_pred_round)
    print('F1: {:.3f}'.format(f1))
    f1w = f1_score(y_true=y_true, y_pred=y_pred_round, average='weighted')
    print('\tF1 (weighted): {:.3f}'.format(f1w))
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    print('ROC AUC: {:.3f}'.format(roc_auc))
    y_pred = list(y_pred)
    print('% 1: ', sum(y_pred_round) / len(y_pred_round))
    with open(REPORT_FILENAME, 'a') as f:
        f.write('\n{}\t{}\t{}\t{}\t\r\n'.format(acc, f1, f1w, roc_auc))
    return acc, f1, f1w, roc_auc


def get_lstm_model(NUM_WORDS, EMBEDDING_DIM, embedding_matrix, lstm_dim=32, trainable_embeddings=False):
    K.clear_session()
    inp = Input(shape=(None, MAX_LENGTH))
    embedding_layer = Embedding(NUM_WORDS + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=trainable_embeddings)
    embedding = TimeDistributed(embedding_layer)(inp)
    lstm_out = TimeDistributed(LSTM(lstm_dim, return_sequences=False))(embedding)
    lstm_out = BatchNormalization()(lstm_out)
    s = LSTM(lstm_dim, return_sequences=False)(lstm_out)
    s = BatchNormalization()(s)
    x = Dense(1, activation='sigmoid')(s)

    seed_everything(SEED)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='nadam')
    model.summary()
    return model


def train_lstm_model(model, train, tokenizer, num_epochs=5):
    t = tqdm(train.iterrows(), total=train.shape[0])
    for epoch in range(num_epochs):
        for idx, row in t:
            x_train = tokenizer.texts_to_sequences(row['text'])
            x_train = pad_sequences(x_train, maxlen=MAX_LENGTH, value=0.0)
            y_train = np.array([int(row['target'])])
            loss = model.train_on_batch(x_train.reshape(1, -1, MAX_LENGTH),
                                        y=y_train)  # reshape input to be 3D [samples, timesteps, features]
            t.set_description('ML (loss=%g)' % loss)
    return model


def predict_lstm_model(model, val, tokenizer):
    pred = [None] * len(val)
    cntr = 0
    for idx, row in tqdm(val.iterrows(), total=val.shape[0]):
        x_test = tokenizer.texts_to_sequences(row['text'])
        x_test = pad_sequences(x_test, maxlen=MAX_LENGTH, value=0.0)
        pred[cntr] = model.predict_on_batch(x_test.reshape(1, -1, MAX_LENGTH)).item()
        cntr += 1
    return pred


def scale_features(train, val, scaler_type, features=None):
    if features is None:
        features = train.columns
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train[features])
    val_x = scaler.transform(val[features].values)
    return train_x, val_x


def get_mlp_model(num_units=32, input_dim=4, num_layers=2, rate=0.3):
    K.clear_session()
    model = Sequential()
    for _ in range(num_layers):
        model.add(Dense(num_units, activation='selu', input_dim=input_dim))
        model.add(BatchNormalization())
        model.add(AlphaDropout(rate))
    model.add(Dense(1, activation='sigmoid'))

    seed_everything(SEED)

    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.summary()
    return model


def train_mlp(model, train_x, train_y, val_x, val_y, class_weight_, model_name='', epochs=50, batch_size=128):
    es = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, verbose=1)
    model_filename = 'reg_crypto_reddit_mlp' + model_name + '.h5'
    ch = ModelCheckpoint(save_best_only=True, filepath=model_filename, monitor='val_loss', mode='min')
    model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
              callbacks=[es, reduce_lr, ch], class_weight=class_weight_, verbose=0)
    model.load_weights(model_filename)
    model.evaluate(val_x, val_y)
    return model


def get_hybrid_model(NUM_WORDS, EMBEDDING_DIM, embedding_matrix, lstm_dim=32, dense_dim=32, input_dim=4,
                     trainable_embeddings=False):
    K.clear_session()
    inp1 = Input(shape=(None, MAX_LENGTH))
    inp2 = Input(shape=(input_dim,))
    embedding_layer = Embedding(NUM_WORDS + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                trainable=trainable_embeddings)
    embedding = TimeDistributed(embedding_layer)(inp1)
    lstm_out = TimeDistributed(LSTM(lstm_dim, return_sequences=False))(embedding)
    s1 = LSTM(lstm_dim, return_sequences=False)(lstm_out)
    s1 = BatchNormalization()(s1)
    s2 = Dense(dense_dim, activation='selu')(inp2)
    s2 = BatchNormalization()(s2)
    s = Concatenate()([s1, s2])
    x = Dense(1, activation='sigmoid')(s)

    seed_everything(SEED)

    combined_model = Model(inputs=[inp1, inp2], outputs=x)
    combined_model.compile(loss='binary_crossentropy', optimizer='nadam')
    combined_model.summary()
    return combined_model


def train_hybrid_model(train, tokenizer, model, features, num_epochs=5, verbose=0):
    t = tqdm(train.iterrows(), total=train.shape[0])
    for epoch in range(num_epochs):
        for idx, row in t:
            x_train = tokenizer.texts_to_sequences(row['text'])
            x_train = pad_sequences(x_train, maxlen=MAX_LENGTH, value=0.0)
            y_train = np.array([int(row['target'])])
            loss = model.train_on_batch([x_train.reshape(1, -1, MAX_LENGTH), np.array([row[features].values])],
                                        y=y_train)
            if verbose > 0: t.set_description('ML (loss=%g)' % loss)
    return model


def predict_hybrid_model(val, tokenizer, model, features):
    pred = [None] * len(val)
    cntr = 0
    for idx, row in tqdm(val.iterrows(), total=val.shape[0]):
        x_test = tokenizer.texts_to_sequences(row['text'])
        x_test = pad_sequences(x_test, maxlen=MAX_LENGTH, value=0.0)
        y_test = np.array([int(row['target'])])
        pred[cntr] = model.predict_on_batch([x_test.reshape(1, -1, MAX_LENGTH), np.array([row[features].values])])
        cntr += 1
    return pred


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def prepare_lstm_sequence(data_x, data_y):
    in_seqs = [None] * data_x.shape[1]
    for i in range(data_x.shape[1]):
        tmp = array(data_x[:, i])
        tmp = tmp.reshape((len(tmp), 1))
        in_seqs[i] = tmp
        del tmp
    out_seq = array(data_y.astype(int))
    out_seq = out_seq.reshape((len(out_seq), 1))
    # horizontally stack columns
    dataset = hstack(in_seqs + [out_seq])
    return dataset


def get_lstm_2_model(n_steps, n_features, lstm_dim=50, rate=0.3):
    K.clear_session()
    model = Sequential()
    model.add(LSTM(lstm_dim, input_shape=(n_steps, n_features), recurrent_dropout=rate))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))

    seed_everything(SEED)

    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def predict_lstm_2_model(val_y, model, val_x, n_steps, n_features):
    y_pred = [0] * len(val_y)
    for idx in range(len(val_y)):
        try:
            x_input = val_x[idx - n_steps:idx, :]
            x_input = x_input.reshape((1, n_steps, n_features))
            yhat = model.predict(x_input, verbose=0)
            y_pred[idx] = yhat.item()
        except:
            continue
    y_pred = np.round(y_pred)
    return y_pred


def get_mlp_bow_model(num_features=4, dense_dim=32):
    K.clear_session()
    model = Sequential()
    model.add(Dense(dense_dim, activation='selu', input_dim=num_features))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))

    seed_everything(SEED)

    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.summary()
    return model


@ex.config
def cfg():
    basetable_filename = "../../data/processed/reddit/new_basetable_nonan.p"
    model_type = 'nb'
    tuning_mode = 'rand'
    standardize = True
    train_test_prop = 0.8
    text_features = 'none'
    time_range = None
    coin_name = 'BTC'
    use_smote = False
    max_lag = 7
    score_metric = 'roc_auc'
    save = False
    all_comb = False
    next_day = True

    model_name = '_'.join(
        ['crypto', determine_data_source(basetable_filename), model_type, tuning_mode, str(standardize), text_features,
         score_metric])

    print('Basetable:', basetable_filename)
    print('Model:', model_type, '| Train/test:', train_test_prop, '| Text features:',
          text_features, '| Tuning mode:', tuning_mode)
    if time_range is not None:
        if time_range == 'default':
            time_range = [pd.Timestamp.strptime("01/01/2017", "%d/%m/%Y"),
                          pd.Timestamp.strptime("30/11/2017", "%d/%m/%Y")]
        else:
            start_time, end_time = time_range.split('-')
            time_range = [pd.Timestamp.strptime(start_time, "%d/%m/%Y"), pd.Timestamp.strptime(end_time, "%d/%m/%Y")]
    else:
        time_range = []


def load_data(filepath, time_range=[], coin='BTC', next_day=True, model_type='nb'):
    data = pickle.load(open(filepath, 'rb'))
    data.rename({'text_polarity_len': 'len'}, axis='columns', inplace=True)
    if coin != '':
        print('Filter by cryptocurrency coin:', coin)
        data = data[data.coin == coin]
    if next_day:
        data['next_day_return'] = data['return'].shift(-1)
        data['direction'] = data['next_day_return'].apply(lambda x: x >= 0)
        data.drop('next_day_return', axis=1, inplace=True)
    # data = data[(data.index >= pd.Timestamp('2015-09-15')) & (data.index <= pd.Timestamp('2019-02-18'))]
    if len(time_range) > 0:
        print('Filter by time range:', time_range)
        data = data.loc[(data.index >= time_range[0]) & (data.index <= time_range[1])]
    len_data = len(data)
    data = data.dropna()
    print('Dropped NaN:', len_data, '->', len(data))
    data = data.sort_index()
    if 'hybrid' in model_type or 'text' in model_type:
        comments = pickle.load(open('/'.join(filepath.split('/')[:-1]) + '/unique_comments_text-cleaner_all.p', 'rb'))
        timestamp_column = list(set(['date', 'published_on', 'short_date']).intersection(set(comments.columns)))[0]
        text_column = list(set(['text', 'body']).intersection(set(comments.columns)))[0]
        comments['date'] = comments[timestamp_column].apply(pd.Timestamp)
        month_day = comments.date.map(lambda x: str(x.year) + '-' + str(x.month) + '-' + str(x.day))
        agg_comments = pd.pivot_table(comments,
                                      values=[c for c in comments.columns if text_column == c],
                                      index=[month_day],
                                      aggfunc={c: list for c in comments.columns if text_column == c},
                                      fill_value=0)
        agg_comments.columns = [''.join(col) for col in agg_comments.columns]
        agg_comments = agg_comments.reset_index()
        agg_comments = agg_comments.set_index('date')
        agg_comments.index = pd.to_datetime(agg_comments.index)
        agg_comments.sort_index(inplace=True, ascending=True)
        df = agg_comments.copy(deep=True)
        df = df.loc[~df.index.duplicated(keep='first')]
        df = df.join(data)
        df = df.dropna()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.rename({'body': 'text'}, axis=1, inplace=True)
        return df
    return data


def pre_process(data, text_features, max_lag=3, rolling_window=1):
    if 'len' in data.columns:
        for col in data.columns:
            data.loc[((data.index < '2017-07-11') | (data.index > '2017-07-23')) & data[col].isna(), 'len'] = 0

    data = data.replace(np.inf, 0)

    Y = data['direction']
    Y = Y.astype(bool)

    data = data.interpolate()

    ma_cols = ['interest'] + [c for c in text_features if c != 'text']
    ma = data[data.columns.intersection(ma_cols)].rolling(rolling_window).mean()
    non_senti = data[data.columns.difference(ma_cols)]
    data = pd.concat([non_senti, ma], axis=1)

    non_diff = ['return', 'volumefrom'] + [t for t in text_features if 'tot_posneg' in t or t == 'text']
    diff = ['volumeto', 'spread', 'interest'] + [c for c in text_features if c != 'text']
    data_nd = data[data.columns.intersection(non_diff)]
    data_d = data[data.columns.intersection(diff)].diff()
    data = pd.concat([data_nd, data_d], axis=1, sort=False)
    data = data.dropna()

    # Create lagging variables
    print("Lagging variables, max_lag:", max_lag)
    lags = max_lag
    X = pd.DataFrame()
    for i in range(lags):
        lag = data.shift(i + 1).add_suffix('_' + str(i + 1))
        X = pd.concat([X, lag], axis=1)
    if 'text' in text_features:
        X = X.join(data['text'])
    X = X.dropna()
    y = Y.loc[Y.index.isin(X.index)]
    return X, y


@ex.capture
def track_score(_run, score_type, score_value):
    _run.log_scalar(score_type, score_value)


def train_model(X, y, model, model_type, param_grid, train_test_prop=0.8, n_splits=5, standardize=True,
                score_metric='roc_auc',
                tuning='rand', use_smote=False, n_steps=1):
    # Split into train and test using nested time-series cross-validation
    # Use SMOTE and class weights to handle class imbalance.
    seed_everything(SEED)
    cv_split = int(len(X) * train_test_prop)
    X_train, X_test = X.iloc[:cv_split], X.iloc[cv_split:]
    y_train, y_test = y.iloc[:cv_split], y.iloc[cv_split:]
    print('Train size:', len(X_train), len(y_train), 'test size:', len(X_test), len(y_test))
    print(X_train.head())
    print(X_test.head())
    if 'text' in model_type or model_type == 'neural_hybrid':
        use_smote = False
    if use_smote:
        xc = X_train.columns
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_train, columns=xc)
        y_train = pd.Series(y_train)
        print('Applied SMOTE. New class balance:', sum(y_train) / len(y_train), sum(y_test) / len(y_test))
        print('Train size:', len(X_train), len(y_train))

    tscv = TimeSeriesSplit(n_splits=n_splits)

    if 'neural' in model_type:
        # text_features = [c for c in X.columns if
        #                  ('text' in c or 'topic' in c or c == 'len' or 'score' in c) and (c != 'text')]
        financial_features = [c for c in X.columns if
                              any(ext in c for ext in ['close', 'volumefrom', 'volumeto', 'spread', 'return'])]
        trend_features = [c for c in X.columns if 'interest' in c]
        if 'text' in model_type or 'hybrid' in model_type:
            tokenizer = Tokenizer(oov_token='#OOV#')
            tokenizer.fit_on_texts(X_train.text.values)
            vocabulary = tokenizer.word_index
            word_vectors = load_embeddings()
            embedding_dim = 50
            num_words = len(vocabulary)
            embedding_matrix = np.zeros((num_words + 1, embedding_dim))
            overflow_cntr = 0
            oov_cntr = 0
            for word, i in vocabulary.items():
                if i > num_words:
                    print(i)
                    overflow_cntr += 1
                    continue
                try:
                    embedding_vector = word_vectors[word]
                    embedding_matrix[i] = embedding_vector
                except KeyError:
                    embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), embedding_dim)
                    oov_cntr += 1
        start_time = time.time()
        results = {'mean_test_accuracy': [], 'mean_test_roc_auc': [], 'mean_test_f1_weighted': [],
                   'mean_test_disbalance': []}
        for train_index, test_index in tscv.split(X_train):
            X_cv_train, X_cv_test = X_train.iloc[train_index], X_train.iloc[test_index]
            y_cv_train, y_cv_test = y_train.iloc[train_index], y_train.iloc[test_index]
            print(len(X_cv_train), len(y_cv_train), len(X_cv_test), len(y_cv_test))
            class_weights = class_weight.compute_class_weight('balanced',
                                                              np.unique(y_cv_train),
                                                              y_cv_train)
            class_weight_ = dict(enumerate(class_weights))
            es = EarlyStopping(monitor='val_loss', patience=10)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=5, verbose=1)

            if model_type == 'neural_lstm_text':
                model = get_lstm_model(embedding_dim, num_words, embedding_matrix)
                model = train_lstm_model(model, X_cv_train, tokenizer)
                pred = predict_lstm_model(model, X_cv_test, tokenizer)
            elif model_type == 'neural_mlp':
                train_x, val_x = scale_features(X_cv_train, X_cv_test, 'standard')
                model = get_mlp_model(input_dim=train_x.shape[1])
                model = train_mlp(model, train_x, y_cv_train, val_x, y_cv_test, class_weight_,
                                  model_name='_standardscaler')
                pred = model.predict(val_x)
            elif model_type == 'neural_hybrid':
                model = get_hybrid_model(num_words, embedding_dim, embedding_matrix, 32, 32, input_dim=4)
                model = train_hybrid_model(X_cv_train, tokenizer, model, financial_features + trend_features)
                pred = predict_hybrid_model(X_cv_test, tokenizer, model, financial_features + trend_features)
                pred = [z for x in pred for y in x for z in y]
            elif model_type == 'neural_lstm':
                train_x, val_x = scale_features(X_cv_train, X_cv_test, 'standard')
                train_dataset = prepare_lstm_sequence(train_x, y_cv_train)
                val_dataset = prepare_lstm_sequence(val_x, y_cv_test)
                n_steps = n_steps
                X_sequences, y_sequences = split_sequences(train_dataset, n_steps)
                X_val_sequences, y_val_sequences = split_sequences(val_dataset, n_steps)
                n_features = X_sequences.shape[2]
                print('val_x', val_x.shape)
                print('val_dataset', val_dataset.shape)
                print('X_val_sequences', X_val_sequences.shape)
                model = get_lstm_2_model(n_steps, n_features)
                model_filename = 'reg_crypto_reddit_lstm_financial_minmax.h5'
                ch = ModelCheckpoint(save_best_only=True, filepath=model_filename, monitor='val_loss', mode='min')
                model.fit(X_sequences, y_sequences, validation_data=(X_val_sequences, y_val_sequences), epochs=45,
                          verbose=0, class_weight=class_weight_, callbacks=[es, reduce_lr, ch])
                model.load_weights(model_filename)
                pred = predict_lstm_2_model(y_cv_test, model, X_val_sequences, n_steps, n_features)
            elif model_type == 'neural_mlp_text':
                vectorizer = CountVectorizer()
                X_train_bow = vectorizer.fit_transform([' '.join(x) for x in X_cv_train.text.values])
                X_val_bow = vectorizer.transform([' '.join(x) for x in X_cv_test.text.values])
                model = get_mlp_bow_model(num_features=X_train_bow.shape[1])
                model_filename = 'reg_crypto_reddit_mlp_bow.h5'
                ch = ModelCheckpoint(save_best_only=True, filepath=model_filename, monitor='val_loss', mode='min')
                model.fit(X_train_bow, y_cv_train, validation_data=(X_val_bow, y_cv_test), epochs=50, batch_size=128,
                          class_weight=class_weight_, callbacks=[es, reduce_lr, ch], verbose=0)
                model.load_weights(model_filename)
                model.evaluate(X_val_bow, y_test)
                pred = model.predict(X_val_bow)

            class_pred = np.round(pred)
            results['mean_test_disbalance'] = results['mean_test_disbalance'] + [
                calculate_disbalance(y_cv_test, class_pred)]
            results['mean_test_accuracy'] = results['mean_test_accuracy'] + [
                metrics.accuracy_score(y_cv_test, class_pred)]
            results['mean_test_roc_auc'] = results['mean_test_roc_auc'] + [metrics.roc_auc_score(y_cv_test, pred)]
            results['mean_test_f1_weighted'] = results['mean_test_f1_weighted'] + [
                metrics.f1_score(y_cv_test, class_pred, average='weighted')]
        elapsed_time = time.time() - start_time
        print('Finished training. Time:', elapsed_time)
        best_param = None
        clf = model
    else:
        # Standardize data.
        if standardize:
            pipe = Pipeline([
                ('scale', StandardScaler()),
                ('clf', model)
            ])
            print("Scaled data (standardized).")
        else:
            pipe = Pipeline([
                ('clf', model)
            ])

        if model_type == 'nb':
            print('Started training.')
            start_time = time.time()
            results = {'mean_test_accuracy': [], 'mean_test_roc_auc': [], 'mean_test_f1_weighted': [],
                       'mean_test_disbalance': []}
            for train_index, test_index in tscv.split(X_train):
                X_cv_train, X_cv_test = X_train.iloc[train_index], X_train.iloc[test_index]
                y_cv_train, y_cv_test = y_train.iloc[train_index], y_train.iloc[test_index]
                class_weights = class_weight.compute_class_weight('balanced',
                                                                  np.unique(y_cv_train),
                                                                  y_cv_train)
                class_weight_ = dict(enumerate(class_weights))
                model.class_weight = class_weight_
                model.fit(X_cv_train, y_cv_train)
                pred = model.predict_proba(X_cv_test)[:, 1]
                class_pred = np.round(pred)
                results['mean_test_disbalance'] = results['mean_test_disbalance'] + [
                    calculate_disbalance(y_cv_test, pred)]
                results['mean_test_accuracy'] = results['mean_test_accuracy'] + [model.score(X_cv_test, y_cv_test)]
                results['mean_test_roc_auc'] = results['mean_test_roc_auc'] + [metrics.roc_auc_score(y_cv_test, pred)]
                results['mean_test_f1_weighted'] = results['mean_test_f1_weighted'] + [
                    metrics.f1_score(y_cv_test, class_pred, average='weighted')]
            elapsed_time = time.time() - start_time
            print('Finished training. Time:', elapsed_time)
            best_param = None
            clf = model
        else:
            class_weights = class_weight.compute_class_weight('balanced',
                                                              np.unique(y),
                                                              y)
            class_weight_ = dict(enumerate(class_weights))
            model.class_weight = class_weight_
            scoring = {'accuracy': 'accuracy', 'roc_auc': 'roc_auc', 'f1_weighted': 'f1_weighted',
                       'disbalance': make_scorer(calculate_disbalance)}
            if tuning == 'rand':
                grid_model = RandomizedSearchCV(estimator=pipe,
                                                param_distributions=param_grid,
                                                n_iter=300,
                                                cv=tscv,
                                                refit=score_metric,
                                                scoring=scoring,
                                                return_train_score=True, verbose=1, n_jobs=-1)
            else:
                grid_model = GridSearchCV(estimator=pipe,
                                          param_grid=param_grid,
                                          cv=tscv,
                                          scoring=scoring,
                                          refit=score_metric,
                                          return_train_score=True, verbose=1, n_jobs=-1)

            print('Started training.')
            start_time = time.time()
            grid_model.fit(X_train, y_train)
            best_param = grid_model.best_params_
            results = grid_model.cv_results_
            elapsed_time = time.time() - start_time
            print('Finished training. Time:', elapsed_time)
            clf = grid_model.best_estimator_
    validation_accuracy = np.mean(results['mean_test_accuracy'])
    validation_roc_auc = np.mean(results['mean_test_roc_auc'])
    validation_f1_weighted = np.mean(results['mean_test_f1_weighted'])
    validation_disbalance = np.mean(results['mean_test_disbalance'])
    print('Prediction performance on validation set:')
    print('\tAccuracy:', np.round(validation_accuracy, 3))
    print('\tROC AUC score:', np.round(validation_roc_auc, 3))
    print('\tF1 (weighted) score:', np.round(validation_f1_weighted, 3))
    print('\t%1:', np.round(validation_disbalance, 3))
    if model_type == 'rf' or model_type == 'xgboost':
        md = SelectFromModel(clf.named_steps['clf'], prefit=True)
        names = X.iloc[:, md.get_support()].columns.values
        feat_imp = (clf.named_steps['clf'].feature_importances_, names)
    else:
        feat_imp = None
    print('Feature importance', feat_imp)

    pred_results = {'Model': clf,
                    'Feature importances': feat_imp,
                    'Parameter value': best_param,
                    'Prediction accuracy': validation_accuracy,
                    'AUC score': validation_roc_auc,
                    'F1 score': validation_f1_weighted,
                    '%1': validation_disbalance}
    return pred_results


@ex.automain
def run(model_type, tuning_mode, standardize, train_test_prop, text_features, time_range,
        score_metric, save, model_name, basetable_filename, max_lag, coin_name, use_smote, next_day):
    print('Training an sklearn model with sacred support.')
    data = load_data(basetable_filename, time_range, coin_name, next_day, model_type)
    text_features_polarity = ['text_polarity_negative',
                              'text_polarity_positive', 'text_polarity_sum',
                              'text_polarity_tot_neg', 'text_polarity_tot_pos',
                              'text_polarity_tot_posneg', 'text_polarity_posneg',
                              'sentiment_scores_0_mean', 'sentiment_scores_1_mean', 'sentiment_scores_2_mean'
                              'sentiment0_mean', 'sentiment1_mean', 'sentiment2_mean']
    text_features_subjectivity = ['text_subjectivity_negative', 'text_subjectivity_positive',
                                  'text_subjectivity_sum', 'text_subjectivity_tot_neg', 'text_subjectivity_tot_posneg',
                                  'text_subjectivity_posneg', 'text_subjectivity_tot_pos']
    text_features_topic = ['topic' + str(i) + '_mean' for i in range(0, 15)] + ['topic_scores_' + str(i) + '_mean' for i
                                                                                in range(0, 15)]
    other_text_features = ['len', 'text']
    all_text_features = other_text_features + text_features_topic + text_features_polarity + text_features_subjectivity
    selected_text_features = other_text_features
    remove_not_text = False
    if text_features is not None:
        if 'only' in text_features:
            if text_features == 'only':
                selected_text_features = text_features_polarity + text_features_subjectivity + other_text_features + text_features_topic
            remove_not_text = True
        if text_features == 'all':
            selected_text_features = text_features_polarity + text_features_subjectivity + other_text_features + text_features_topic
        elif text_features == 'none':
            selected_text_features = []
        else:
            if 'polarity' in text_features:
                selected_text_features = selected_text_features + text_features_polarity
            if 'subjectivity' in text_features:
                selected_text_features = selected_text_features + text_features_subjectivity
            if 'topic' in text_features:
                selected_text_features = selected_text_features + text_features_topic
    used_text_features = list(set(selected_text_features).intersection(set(data.columns)))
    if len(selected_text_features) > len(used_text_features):
        print(
            'WARNING: not all text features used, because the corresponding columns are not in the basetable dataframe. 2 main reasons: wrong column\'s names or these features were not extracted for this dataset.')
        print('Omitted columns:', list(set(selected_text_features) - set(used_text_features)))
    data = data.drop(list(set(all_text_features).intersection(set(data.columns)) - set(used_text_features)), axis=1)
    if remove_not_text:
        data.drop(columns=['return', 'close', 'volumefrom', 'volumeto', 'spread', 'interest'], inplace=True)

    print("Used text features:", used_text_features)
    print('Features:', data.columns)
    X, y = pre_process(data, used_text_features, max_lag=max_lag)
    print(X.head())
    print('Dataset size:', len(X), 'from', min(data.index), 'to', max(data.index), '| Target class balance:',
          np.round(sum(y) / len(y), 3))
    n_alphas = 1000

    random_state = 456
    param_grid = {}
    clf = None
    if model_type is not None and 'neural' not in model_type:
        if model_type == 'svm':
            n_alphas = 100
            clf = SVC(random_state=random_state, kernel='rbf')
            param_grid = {'clf__C': np.logspace(-3, 3, n_alphas)}
        elif model_type == 'nb':
            clf = GaussianNB()
        elif model_type == 'logreg':
            clf = LogisticRegression(random_state=random_state, solver='liblinear')
            param_grid = {'clf__C': np.logspace(-10, 10, n_alphas)}
        elif model_type == 'rf':
            clf = RandomForestClassifier(random_state=random_state, n_estimators=50)
            if tuning_mode == 'rand':
                param_grid = {"clf__max_depth": sp_randint(1, 25),
                              "clf__min_samples_leaf": sp_randint(2, 50),
                              "clf__min_samples_split": sp_randint(2, 56),
                              "clf__max_features": sp_randint(1, X.shape[1])}
            else:
                param_grid = {"clf__max_depth": np.arange(1, 25, 2),
                              "clf__min_samples_leaf": np.arange(2, 50, 2),
                              "clf__min_samples_split": np.arange(2, 56, 2),
                              "clf__max_features": np.arange(1, X.shape[1], 1)}
        elif model_type == 'xgboost':
            clf = GradientBoostingClassifier(random_state=random_state, n_estimators=50)
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

    pred_results = train_model(X, y, clf, model_type, train_test_prop=train_test_prop, param_grid=param_grid,
                               tuning=tuning_mode, score_metric=score_metric,
                               standardize=standardize, use_smote=use_smote)
    print(pred_results)
    if save:
        dirname = os.path.dirname(__file__)
        filepath = os.path.join(dirname, '../../models/classification/2021/')
        pickle.dump(pred_results, open(filepath + model_name + '.p', 'wb'))
    return pred_results['Prediction accuracy']

# python TrainSacredModel.py with basetable_filename=../../data/processed/bitcointalk/date_no_structure/new_basetable_nonan.p
# python TrainSacredModel.py -m sacred_sep19 with basetable_filename=../../data/processed/reddit/long/new_basetable_nonan.p text_features=all
# python TrainSacredModel.py with basetable_filename=../../data/processed/bitcointalk/date_no_structure/JST_new_basetable.p model_type=logreg tuning_mode=rand text_features=all train_test_prop=0.8
# python TrainSacredModel.py -m sacred_feb21 with basetable_filename=../../data/processed/reddit/long/new_basetable_nonan.p text_features=all