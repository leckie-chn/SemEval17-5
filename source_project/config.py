import os
import socket

machine = socket.gethostname()

ROOT_DIR = os.path.dirname(__file__)
print(ROOT_DIR)

DATA_DIR = '/home/niyan/SemEval17-05-kar/data'
DUMPED_VECTOR_DIR_HL = os.path.join(DATA_DIR, 'vectors_hl_new/')
DUMPED_VECTOR_DIR = os.path.join(DATA_DIR, 'vectors_mb_new/')
TRUE_LABELS_PATH = os.path.join(DATA_DIR, 'vectors_mb_new/mb_scores.pkl')
# TRUE_LABELS_PATH = '/raid/data/skar3/semeval/vectors/headline_scores.pkl'

PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'processed_data')
RESULTS_DIR = os.path.join(DATA_DIR, 'outputs/')
WORD_EMBEDDING_VECTOR_PATH = '/home/niyan/stockNN/data/wemb/GoogleNews-vectors-negative300.bin'
VOCABULARY_WORD_TO_INDEX_LIST = os.path.join(DATA_DIR, 'preprocessed/headline_vocabulary_to_index.json')
VOCABULARY_INDEX_TO_WORD_LIST = os.path.join(DATA_DIR, 'preprocessed/headline_index_to_word.json')

XDING_SOURCE_PATH = '/home/niyan/stockNN/data/news'

features_to_extract = [
    'rf_unigram',
    'rf_bigram',
    'rf_trigram',
    'unigram',
    'bigram',
    'trigram',
    'binary_unigram',
    'binary_bigram',
    'binary_trigram',
    'char_tri',
    'char_4_gram',
    'char_5_gram',
    # 'two_skip_3_grams',
    # 'two_skip_2_grams',
    'concepts',
    'stemmed_concepts',
    'google_word_emb',
    'cashtag',
    'source',
    'polarity',
    'sensitivity',
    'attention',
    'aptitude',
    'pleasantness',
    'stemmed_polarity',
    'stemmed_sensitivity',
    'stemmed_attention',
    'stemmed_aptitude',
    'stemmed_pleasantness',
    # 'company'
    'scores',
]

features_to_use = [
    'unigram',
    'bigram',
    'trigram',
    'binary_unigram',
    'binary_bigram',
    'binary_trigram',
    'char_tri',
    'char_4_gram',
    'char_5_gram',
    'two_skip_3_grams',
    'two_skip_2_grams',
    'concepts',
    'stemmed_concepts',
    'google_word_emb',
    'polarity',
    'sensitivity',
    'attention',
    'aptitude',
    'pleasantness',
    'stemmed_polarity',
    'stemmed_sensitivity',
    'stemmed_attention',
    'stemmed_aptitude',
    'stemmed_pleasantness',
    'company'
]

features_to_use_mb = [
    'unigram',
    'bigram',
    'trigram',
    'binary_unigram',
    'binary_bigram',
    'binary_trigram',
    # 'char_tri',
    # 'char_4_gram',
    # 'char_5_gram',
    # 'two_skip_3_grams',
    # 'two_skip_2_grams',
    # 'concepts',
    # 'stemmed_concepts',
    'google_word_emb',
    # 'cashtag',
    # 'source',
    'polarity',
    'sensitivity',
    'attention',
    'aptitude',
    'pleasantness',
    'stemmed_polarity',
    'stemmed_sensitivity',
    'stemmed_attention',
    'stemmed_aptitude',
    'stemmed_pleasantness',
]
