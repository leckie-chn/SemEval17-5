# -*- coding: utf-8 -*-
from __future__ import division, print_function

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

__all__ = ['NGramTfidfVectorizer', 'NGramRFVectorizer']


class NGramTfidfVectorizer(TfidfVectorizer):
    """Convert a collection of  documents objects to a matrix of TF-IDF features.

      Refer to super class documentation for further information
    """

    def build_analyzer(self):
        """Overrides the super class method

        Parameter
        ----------
        self

        Returns
        ----------
        analyzer : function
            extract content from document object and then applies analyzer

        """
        analyzer = super(TfidfVectorizer,
                         self).build_analyzer()
        return lambda doc: (w for w in analyzer(doc.content))


class NGramRFVectorizer(CountVectorizer):
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64):
        super(NGramRFVectorizer, self).__init__(input, encoding, decode_error, strip_accents, lowercase, preprocessor,
                                                tokenizer, stop_words, token_pattern, ngram_range, analyzer, max_df,
                                                min_df, max_features, vocabulary, binary, dtype)
        self._rf_weight = None

    def build_analyzer(self):
        analyzer = super(NGramRFVectorizer, self).build_analyzer()
        return lambda doc: (w for w in analyzer(doc.content))

    def _fit_rf(self, X, y):
        """
        fit self._rf_weight: rf weight for each n-gram token
        """
        X = (X > 0).toarray()
        rf_pos = np.sum(X[y > 0, :], axis=0, keepdims=False)
        rf_neg = np.sum(X[y < 0, :], axis=0, keepdims=False)
        self._rf_weight = np.maximum(np.log(2.0 + rf_pos / np.maximum(1.0, rf_neg)),
                                     np.log(2.0 + rf_neg / np.maximum(1.0, rf_pos)))

    def fit(self, raw_documents, y=None):
        assert y is not None
        X = super(NGramRFVectorizer, self).fit_transform(raw_documents, y)
        self._fit_rf(X[:y.shape[0], :], y)

    def fit_transform(self, raw_documents, y=None):
        assert y is not None
        X = super(NGramRFVectorizer, self).fit_transform(raw_documents, y)
        self._fit_rf(X[:y.shape[0], :], y)
        return self._rf_weight[np.newaxis, :] * X.toarray()

    def transform(self, raw_documents):
        X = super(NGramRFVectorizer, self).transform(raw_documents)
        return self._rf_weight[np.newaxis, :] * X.toarray()
