import os, sys

sys.path.append('../')

import keras
import tensorflow as tf

from typing import List, Dict

from keras.layers import LeakyReLU, BatchNormalization, Concatenate
from keras.regularizers import l2, l1
from keras.optimizers import Adam
import keras.backend as K
import pandas as pd

from prepare_data.doc_to_sequence_csv import TextConverter
from prepare_data.preprocess_csv import tokenize_csv_file
from experiments.headline_deep import *

def cos_distance(y_true, y_pred):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())

    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return K.mean(y_true * y_pred, axis=-1)


def tf_auc_hl(y_true, y_pred):
    y_true = K.sign(K.sign(y_true) + 1.0)
    y_pred = (y_pred + 1.0) * 0.5
    score, up_opt = tf.metrics.auc(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

def tf_auc_imdb(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def nn_model(embedding_weights: np.ndarray = None) -> (keras.Model, int):

    input_X = Input(shape=(18, ))
    if embedding_weights is None:
        embedding_weights = joblib.load(config.DUMPED_VECTOR_DIR_HL + 'hl_voc_embeddings.pkl')
    embedding_layer = Embedding(embedding_weights.shape[0],
                            embedding_weights.shape[1],
                            input_length=18,
                            weights=[embedding_weights],
                            trainable=False, name='acl_embedding')(input_X)
    # model.add(m_1)
    sub_model_list = []

    for ks in [1, 2, 3 ,4]:
        conv_1d_layer = Conv1D(256, ks, activation='relu', use_bias=False, kernel_regularizer=l2(5e-4), trainable=True,
                               name='cnn_' + str(ks))(embedding_layer)
        # model.add(Conv1D(32, kernel_size, activation='relu', use_bias=False, kernel_regularizer=l2(1e-2)))
        pooled_layer = GlobalMaxPooling1D()(conv_1d_layer)
        sub_model_list.append(pooled_layer)

    concat_layer =  Concatenate()(sub_model_list)
    # model.add(MaxPooling1D(pool_size=16))
    # model.add(Flatten())
    concat_drop_layer = Dropout(rate=0.95)(concat_layer)
    dense_hidden = Dense(units=50, activation='tanh', name='acl_dense_1')(concat_drop_layer)
    score_output = Dense(units=1, activation='tanh', name='acl_dense_2')(dense_hidden)
    model = Model(input_X, score_output)

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf_auc_hl])
    # model.compile(loss='cosine_proximity', optimizer='adam', metrics={'output_a': cosine_similarity})
    # model.compile(loss=compile_cos_sim_theano, optimizer='adam', metrics=[compile_cos_sim_theano])
    print(model.summary())

    return model


def multi_model():
    input_hl = Input(shape=(18,))
    input_imdb = Input(shape=(200,))
    embedding_weights = joblib.load(config.DUMPED_VECTOR_DIR_HL + 'voc_embeddings.pkl')
    embedding_hl = Embedding(embedding_weights.shape[0],
                                embedding_weights.shape[1],
                                input_length=18,
                                weights=[embedding_weights],
                                trainable=False, name='hl_embedding')(input_hl)
    embedding_imdb = Embedding(embedding_weights.shape[0],
                               embedding_weights.shape[1],
                               input_length=200,
                               weights=[embedding_weights],
                               trainable=False, name='imdb_embedding')(input_imdb)
    # model.add(m_1)
    sub_model_hl = []
    sub_model_imdb = []

    for ks in [1, 2, 3, 4]:
        conv_1d = Conv1D(256, ks, activation='relu', use_bias=False, kernel_regularizer=l2(5e-4), trainable=True,
                               name='cnn_' + str(ks))
        conv1d_hl = conv_1d(embedding_hl)
        conv1d_imdb = conv_1d(embedding_imdb)
        # model.add(Conv1D(32, kernel_size, activation='relu', use_bias=False, kernel_regularizer=l2(1e-2)))
        pooled_hl = GlobalMaxPooling1D()(conv1d_hl)
        pooled_imdb = GlobalMaxPooling1D()(conv1d_imdb)
        sub_model_hl.append(pooled_hl)
        sub_model_imdb.append(pooled_imdb)

    concat_hl = Concatenate()(sub_model_hl)
    concat_drop_hl = Dropout(rate=0.95)(concat_hl)
    dense_hl = Dense(units=50, activation='tanh', name='hl_dense')(concat_drop_hl)
    output_hl = Dense(units=1, activation='tanh', name='hl_output')(dense_hl)
    model_hl = Model(input_hl, output_hl)

    model_hl.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf_auc_hl])
    # model.compile(loss='cosine_proximity', optimizer='adam', metrics={'output_a': cosine_similarity})
    # model.compile(loss=compile_cos_sim_theano, optimizer='adam', metrics=[compile_cos_sim_theano])
    print("Headline Model: ")
    print(model_hl.summary())

    concat_imdb = Concatenate()(sub_model_imdb)
    concat_drop_imdb = Dropout(rate=0.3)(concat_imdb)
    dense_imdb = Dense(units=50, activation='tanh', name='imdb_dense')(concat_drop_imdb)
    output_imdb = Dense(units=1, activation='sigmoid', name='imdb_output')(dense_imdb)
    model_imdb = Model(input_imdb, output_imdb)
    model_imdb.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf_auc_imdb])
    print('IMDB Model: ')
    print(model_imdb.summary())
    return model_hl, model_imdb


class FinancialNewsAnalyser:
    def __init__(self):
        self.model = None
        self.text_converter = TextConverter(config.WORD_EMBEDDING_VECTOR_PATH, True, True)
        self.emb_matrix = None

    def fit(self, news_df: pd.DataFrame, batch_size: int, epochs: int, company_alias: Dict = None, verbose: int = 0):
        news_df_prs = tokenize_csv_file(news_df, should_replace_company=True, should_remove_NE=True,
                                        should_remove_numbers=False, company_alias=company_alias)
        self.emb_matrix = self.text_converter.fit(news_df_prs, 'text')
        self.model = nn_model(self.emb_matrix)
        X = self.text_converter.convert(news_df_prs, 'text')
        y = news_df.as_matrix(['sentiment'])
        print('Embedding Matrix Size: {}\nTraining Data Size, X: {}, Y:{}'.format(np.shape(self.emb_matrix),
                                                                                  np.shape(X),
                                                                                  np.shape(y)))
        print('Start Training...')
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def predict(self, news_df: pd.DataFrame, company_alias: Dict = None) -> pd.DataFrame:
        assert self.model is not None, "Error: should invoke FinancialNewsAnalyser.fit/load before predict"

        news_df_prs = tokenize_csv_file(news_df, should_replace_company=True, should_remove_NE=True,
                                        should_remove_numbers=False, company_alias=company_alias)
        X = self.text_converter.convert(news_df_prs, 'text')
        print('Test Data Size: {}'.format(np.shape(X)))
        print('Predicting...')
        y = self.model.predict(X)
        news_df['sentiment'] = y
        return news_df

    def save(self, path):
        self.model.save_weights(os.path.join(path, 'model.h5'))
        joblib.dump(self.emb_matrix, os.path.join(path, 'voc_embedding.pkl'))
        joblib.dump(self.text_converter, os.path.join(path, 'text_converter.pkl'))

    def load(self, path):
        self.text_converter = joblib.load(os.path.join(path, 'text_converter.pkl'))
        self.emb_matrix = joblib.load(os.path.join(path, 'voc_embedding.pkl'))
        self.model = nn_model(self.emb_matrix)
        self.model.load_weights(os.path.join(path, 'model.h5'))
