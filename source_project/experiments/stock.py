import os
import random
import sys

sys.path.append('../')

import pandas as pd
import numpy as np
import talib
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import xgboost as xgb
import operator
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score, precision_recall_curve, accuracy_score
import matplotlib.pyplot as plt
from typing import List, Dict

yf.pdr_override()


def merge_stock_news(stock_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    news_scores_daily = pd.DataFrame.from_dict({
        'sent_avg': news_df.groupby(['date'])['sentiment'].mean(),
        'sent_max': news_df.groupby(['date'])['sentiment'].max(),
        'sent_min': news_df.groupby(['date'])['sentiment'].min(),
        'sent_sum': news_df.groupby(['date'])['sentiment'].sum(),
    })
    merged_df = stock_df.join(news_scores_daily)
    return merged_df


def extract_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    #    stock_df = stock_df.reindex(pd.date_range(stock_df.first_valid_index(), stock_df.last_valid_index()))
    #    mask = stock_df.isna().as_matrix(['Open']).reshape(-1)
    stock_df = stock_df.fillna(method='pad')
    stock_raw = {
        'open': stock_df.as_matrix(['Open']).reshape(-1),
        'high': stock_df.as_matrix(['High']).reshape(-1),
        'low': stock_df.as_matrix(['Low']).reshape(-1),
        'close': stock_df.as_matrix(['Adj Close']).reshape(-1),
        'volume': stock_df.as_matrix(['Volume']).reshape(-1).astype(np.float64),
    }

    feat_data = {
        'Adj Close': stock_raw['close'],
        'OBV': talib.OBV(stock_raw['close'], stock_raw['volume']),
        'Volume': stock_raw['close'],
        'RSI6': talib.RSI(stock_raw['close'], timeperiod=6),
        'RSI12': talib.RSI(stock_raw['close'], timeperiod=12),
        'SMA3': talib.SMA(stock_raw['close'], timeperiod=3),
        'EMA6': talib.EMA(stock_raw['close'], timeperiod=6),
        'EMA12': talib.EMA(stock_raw['close'], timeperiod=12),
        'ATR14': talib.ATR(stock_raw['high'], stock_raw['low'], stock_raw['close'], timeperiod=14),
        'MFI14': talib.MFI(stock_raw['high'], stock_raw['low'], stock_raw['close'], stock_raw['volume'],
                           timeperiod=14),
        'ADX14': talib.ADX(stock_raw['high'], stock_raw['low'], stock_raw['close'], timeperiod=14),
        'ADX20': talib.ADX(stock_raw['high'], stock_raw['low'], stock_raw['close'], timeperiod=20),
        'MOM1': talib.MOM(stock_raw['close'], timeperiod=1),
        'MOM3': talib.MOM(stock_raw['close'], timeperiod=3),
        'CCI12': talib.CCI(stock_raw['high'], stock_raw['low'], stock_raw['close'], timeperiod=12),
        'CCI20': talib.CCI(stock_raw['high'], stock_raw['low'], stock_raw['close'], timeperiod=20),
        'ROCR3': talib.ROCR(stock_raw['close'], timeperiod=3),
        'ROCR12': talib.ROCR(stock_raw['close'], timeperiod=12),
        'outMACD': talib.MACD(stock_raw['close'])[0],
        'outMACDSignal': talib.MACD(stock_raw['close'])[1],
        'outMACDHist': talib.MACD(stock_raw['close'])[2],
        'WILLR': talib.WILLR(stock_raw['high'], stock_raw['low'], stock_raw['close']),
        'TSF10': talib.TSF(stock_raw['close'], timeperiod=10),
        'TSF20': talib.TSF(stock_raw['close'], timeperiod=20),
        'TRIX': talib.TRIX(stock_raw['close']),
        'BBANDSUPPER': talib.BBANDS(stock_raw['close'])[0],
        'BBANDSMIDDLE': talib.BBANDS(stock_raw['close'])[1],
        'BBANDSLOWER': talib.BBANDS(stock_raw['close'])[2],
    }

    for colname, colval in feat_data.items():
        stock_df[colname] = colval
    # stock_df = stock_df.loc[np.logical_not(mask)]
    return stock_df


def df2array(stock_df: pd.DataFrame, X_feats: List[str], y_feat: str, rescale=False):
    dataX = stock_df.as_matrix(X_feats)
    dataY = stock_df.as_matrix([y_feat]).reshape(-1)
    dataY = np.sign(np.sign(dataY) + 1.0)  # float => label

    dataX = dataX[np.isfinite(dataY), :]
    dataY = dataY[np.isfinite(dataY)]

    dataX = np.nan_to_num(dataX)

    if rescale:
        X_mean = np.mean(dataX, axis=0)
        X_std = np.std(dataX, axis=0)
        dataX = (dataX - X_mean[np.newaxis, :]) / X_std[np.newaxis, :]
    return dataX, dataY


def rank_features(df_train: pd.DataFrame, df_val: pd.DataFrame, X_feats: List[str], y_feat):
    X_train, Y_train = df2array(df_train, X_feats, y_feat)
    X_val, Y_val = df2array(df_val, X_feats, y_feat)
    dtrain = xgb.DMatrix(X_train, Y_train, feature_names=X_feats)
    dval = xgb.DMatrix(X_val, Y_val, feature_names=X_feats)

    rank_all = dict(zip(X_feats, [0.0] * len(X_feats)))
    rank_score = []
    for i in range(100):
        param = {
            'objective': 'binary:logistic',
            'eta': 0.01,
            'max_depth': 5,
            'min_child_weight': 5,
            'colsample_bytree': 0.3,
            'subsample': 0.2,
            'gamma': 1.0,
            'metric': 'error',
            'seed': random.randint(0, 65536),
        }

        score = {}
        bst = xgb.train(param, dtrain, 500, evals=[(dtrain, 'train'), (dval, 'val')], verbose_eval=False, evals_result=score)
        best_round, score = min(enumerate(score['val']['error']), key=operator.itemgetter(1))
        bst = xgb.train(param, dtrain, best_round, evals=[(dtrain, 'train'), (dval, 'val')], verbose_eval=False)
        rank = bst.get_score(importance_type='gain')
        rank_score.append((rank, score))

    rank_score = sorted(rank_score, key=operator.itemgetter(1), reverse=False)
    for rank in rank_score[:20]:
        for feat in rank[0].keys():
            rank_all[feat] += rank[0][feat]

    return sorted(rank_all.items(), key=operator.itemgetter(1), reverse=True)


def get_data_yahoo_cached(company: str):
    root_dir = '/home1/stocks/niyan/stockNN/data/stocks'
    dump_path = os.path.join(root_dir, company + '.h5')
    if not os.path.exists(dump_path):
        data_df = pdr.get_data_yahoo(company)  # type: pd.DataFrame
        data_df.to_hdf(dump_path, key=company)
    else:
        data_df = pd.read_hdf(dump_path, key=company)

    return data_df


def prepare_data(company: str, news_df: pd.DataFrame = None):
    stock_df = get_data_yahoo_cached(company)
    sp_df = get_data_yahoo_cached('^GSPC')
    nasdaq_df = get_data_yahoo_cached('^IXIC')
    stock_df = extract_features(stock_df)
    sp_df = extract_features(sp_df)
    nasdaq_df = extract_features(nasdaq_df)
    sp_df = sp_df.rename(dict([(name, 'SP_' + name) for name in sp_df.columns.values.tolist()]), axis='columns')
    nasdaq_df = nasdaq_df.rename(dict([(name, 'N_' + name) for name in nasdaq_df.columns.values.tolist()]),
                                 axis='columns')
    all_df = stock_df.join(sp_df).join(nasdaq_df)

    if news_df is not None:
        all_df = merge_stock_news(all_df, news_df)
    return all_df


def test_main(company: str, news_df: pd.DataFrame = None):
    all_df = prepare_data(company, news_df)

    df_train = all_df.loc['2006-11-18':'2013-11-10']
    ranks = rank_features(df_train)

    gain_all = sum([v[1] for v in ranks])
    gain_sum = 0.0
    feat_used = set()
    print('Features')
    for v in ranks:
        gain_sum += v[1]
        print(v[0], '\t', v[1])
        if gain_sum > gain_all * 0.7:
            print('', end='\t')
            continue
        feat_used.add(v[0])

    if news_df is not None:
        feat_used.add('sent3')
        feat_used.add('sent5')
        feat_used.add('sentiment')

    df_train = all_df.loc['2006-11-18':'2013-01-01']
    X_train, y_train, _, _ = df2array(df_train, feat_list=feat_used, rescale=True)

    model = SVC()
    params = {
        'C': np.power(10.0, np.arange(-1.0, 1.0, 0.1)),
        'kernel': ['rbf'],
        'gamma': np.power(10.0, np.arange(-6.0, -2.0, 0.1)),
        'cache_size': [2000.0],
    }

    model = GridSearchCV(model, params, scoring=make_scorer(roc_auc_score), n_jobs=6, cv=5)
    model.fit(X_train, y_train)

    df_test = all_df.loc['2013-01-02':'2013-03-10']
    X_test, y_test, _, _ = df2array(df_test, feat_list=feat_used, rescale=True)

    return pd.DataFrame(model.cv_results_), model.score(X_test, y_test)


def plot_pr_curve(y_true: np.ndarray, y_pred: np.ndarray):
    average_precision = average_precision_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()


def svm_cv(df_train, df_test, X_feats: List[str], y_feat: str, params: Dict[str, float]):
    X_train, y_train = df2array(df_train, X_feats, y_feat)
    X_val, y_val = df2array(df_test, X_feats, y_feat)
    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)
    y = y * 2 - 1.0
    custom_cv = [(np.arange(0, np.shape(X_train)[0]), np.arange(np.shape(X_train)[0], np.shape(X)[0]))]
    svm_model = SVC()
    model = GridSearchCV(svm_model, params, cv=5, scoring=make_scorer(accuracy_score), n_jobs=6)
    model.fit(X, y)
    result = pd.DataFrame(model.cv_results_).sort_values(by='mean_test_score', ascending=False)
    return model.best_params_, result


def xgb_cv(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, params: Dict[str, float],
           w_train: np.ndarray = None, w_val: np.ndarray = None):
    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)
    w = np.concatenate([w_train, w_val], axis=0) if w_train is not None else None
    custom_cv = [(np.arange(0, np.shape(X_train)[0]), np.arange(np.shape(X_train)[0], np.shape(X)[0]))]
    xgb_model = xgb.XGBClassifier()
    model = GridSearchCV(xgb_model, params, cv=custom_cv, scoring=make_scorer(roc_auc_score))
    model.fit(X, y, eval_metric='auc', early_stopping_rounds=5, eval_set=[(X_train, y_train), (X_val, y_val)],
              sample_weight=w)
    result = pd.DataFrame(model.cv_results_).sort_values(by='mean_test_score', ascending=False)
    return model.best_params_, result


def extrace_target(stock_df: pd.DataFrame, colname: str, periods=5)->pd.DataFrame:
    y_colname = 'y_' + colname + '_' + str(periods)
    stock_df[y_colname] = -1.0 * stock_df.diff(periods=-periods)[colname]
    return stock_df