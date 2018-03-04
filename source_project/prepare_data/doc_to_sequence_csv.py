import sys
import os

sys.path.append('/raid/data/skar3/semeval/source/ml_semeval17/')
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors
import config
import codecs
from sklearn.externals import joblib
import pandas as pd

MAX_NB_WORDS = 5000


def get_vocabulary_size(data):
    words = set()

    for doc in data:
        tokens = doc.split()
        for t in tokens:
            words.add(t)

    print('\n'.join(sorted(words)))
    print(len(words))


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = codecs.open(gloveFile, 'r', encoding='latin-1').read().split('\n')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        # print(word)
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


class TextConverter:
    def __init__(self, embed_source: str, encode_entity=True, encode_numeric=True, sl=None):
        self.tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
        if embed_source.endswith('.bin'):
            self.model = KeyedVectors.load_word2vec_format(embed_source, binary=True, encoding='utf-8')
        else:
            self.model = loadGloveModel(embed_source)
        self.encode_entity = encode_entity
        self.encode_numeric = encode_numeric
        self.wemb_size = 300
        self.embed_dim = self.wemb_size + (1 if encode_entity else 0) + (2 if encode_numeric else 0)
        self.entity_bit = 300
        self.log_bit = 301 if encode_entity else 300
        self.digit_bit = 302 if encode_entity else 301
        self.sl = sl
        self.emb_matrix = None

    def fit(self, df: pd.DataFrame, colname: str):
        self.tokenizer.fit_on_texts(df[colname].values)
        sequences = self.tokenizer.texts_to_sequences(df[colname].values)
        if self.sl is None:
            self.sl = max(list(map(lambda x: len(x), sequences)))
        embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1, self.embed_dim))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = np.zeros(self.embed_dim)
            if self.encode_numeric and str(word).isnumeric():
                x_val = float(word)
                embedding_vector[self.log_bit] = np.log10(x_val)
                embedding_vector[self.digit_bit] = x_val / np.power(10, np.trunc(np.log10(x_val)))
            elif self.encode_entity and str(word).lower() == 'tcompany':
                embedding_vector[self.entity_bit] = 1.0
            elif self.encode_entity and str(word).lower() == 'namedentity':
                embedding_vector[self.entity_bit] = -1.0
            elif word in self.model:
                embedding_vector[:self.wemb_size] = self.model[word]
                # VOC_set.add(word)
            embedding_matrix[i] = embedding_vector
        self.emb_matrix = embedding_matrix
        return self.emb_matrix

    def convert(self, df: pd.DataFrame, colname: str) -> np.ndarray:
        sequences = self.tokenizer.texts_to_sequences(df[colname].values)
        data = pad_sequences(sequences, maxlen=self.sl, padding='post', truncating='post')
        return data

# def convert_into_sequences(df, colname, encode_entity = True, encode_numeric=True):
#     print(sl)
#
#     lens = list(map(lambda s: len(s), sequences))
#     print(len(lens), max(lens))
#
#     index_to_word = {v: k for k, v in tokenizer.word_index.items()}
#     print('Loading Embeddings')
#     model =
#     # model = loadGloveModel(config.WORD_EMBEDDING_VECTOR_PATH)
#
#     print('Embeddings Loaded')
#     OOV_set = set()
#     VOC_set = set()
#
#
#
#     print('Dump hl_sequence.pkl, shape: {}'.format(np.shape(data)))
#     # joblib.dump(data, os.path.join(config.DUMPED_VECTOR_DIR_HL, 'hl_sequences.pkl'))
#     print('Dump hl_voc_embeddings.pkl, shape: {}'.format(np.shape(embedding_matrix)))
#     # joblib.dump(embedding_matrix, os.path.join(config.DUMPED_VECTOR_DIR_HL, 'hl_voc_embeddings.pkl'))
#
#     print('\n\n')
#
#     print('Out of Vocabulary words:')
#     print(OOV_set)
#
#     print('\n\n')
#
#     print('Vocab words: {} in total'.format(len(VOC_set)))
#     print(sorted(VOC_set))
#     return data, embedding_matrix


# def metadata_to_emb_vectors():
#
#     for f in config.features_to_use:
#         transformed_vector = joblib.load(config.DUMPED_VECTOR_DIR + f + '.pkl')


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(config.DATA_DIR, 'raw', 'hl_train_trial_test_new_raw.csv'), encoding='utf-8')
    df['text'] = df['text'].fillna('')
    convert_into_sequences(df, 'text')
    # # get_vocabulary_size(data)
    # convert_into_sequences( data )

    # metadata_to_emb_vectors()
