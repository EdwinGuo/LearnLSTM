from six.moves.urllib.request import urlretrieve
import os
import os.path
import numpy as np
import pickle
from collections import Counter
from itertools import chain
import json
import zipfile
import re

# define variables here
seed = 42
vocab_size = 40000
embedding_dim = 100
lower = False # dont lower case the text

n_bytes = 2**31
max_bytes = 2**31 - 1
d_bytes = bytearray(n_bytes)
glove_data_dir = 'glove_data'
saved_train_data_file = 'data/data_title_content.pkl'
glove_index_dict = {}
globale_scale=.1
glove_thr = 0.5

np.random.seed(seed)

# training data's headers and actual contents
title = []
content = []

empty = 0 # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos+1 # first real word

    def get_num_line(glove_file):
        # helper function here
        glove_n_symbols = sum(1 for line in open(glove_file))
        return glove_n_symbols

    def get_vocab(lst):
        vocabcount = Counter(w for txt in lst for w in txt.split())
        vocab = list(map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1])))
        return vocab, vocabcount

# return a matrix of shape (num_of_line_in_glove_file, emdedding_dim)
def retrieve_glove_embedding_weight(glove_file):
    glove_n_symbols = get_num_line(glove_file)
    glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))

    with open(glove_file, 'r') as fp:
        i = 0
        for l in fp:
            l = l.strip().split()
            w = l[0]
            glove_index_dict[w.lower()] = i
            glove_embedding_weights[i,:] = l[1:]
            i += 1

    glove_embedding_weights *= globale_scale
    return glove_embedding_weights

# download file from the given url
def maybe_download(url, filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(glove_data_dir)
                print('Found and verified %s' % filename)
        else:
            print(statinfo.st_size)
            raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
        return filename

# data is from http://research.signalmedia.co/newsir16/signal-dataset.html
# load the data into title and content (only use one percent from the training
# data) only call this function when
def prepare_training_data(initial_train_data_file, train_data_file_pickle):
    titles = []
    contents = []
    if not os.path.exists(train_data_file_pickle):
        for data_json in open(initial_train_data_file, 'r'):
            d = json.loads(data_json)
            titles.append(d.get('title'))
            contents.append(d.get('content'))

                    bytes_out = pickle.dumps((titles, contents))
                    with open(train_data_file_pickle, 'wb') as f_out:
                        for idx in range(0, n_bytes, max_bytes):
                            f_out.write(bytes_out[idx:idx+max_bytes])

        f_out.close()
    else:
        bytes_in = bytearray(0)
        input_size = os.path.getsize(train_data_file_pickle)
        with open(train_data_file_pickle, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
                titles, contents = pickle.loads(bytes_in)

    f_in.close()

    return titles, contents

    def get_idx(vocab, vocabcount):
        word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
        word2idx['<empty>'] = empty
        word2idx['<eos>'] = eos
        idx2word = dict((idx,word) for word,idx in word2idx.items())
        return word2idx, idx2word

        def create_own_vocab_embedding(glove_embedding_weights, idx2word):
            shape = (vocab_size, embedding_dim)
            scale = glove_embedding_weights.std()*np.sqrt(12)/2
            embedding = np.random.uniform(low=-scale, high=scale, size=shape)
            c = 0
            for i in range(vocab_size, idx2word):
                w = idx2word[i]
                g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
                if g is None and w.startswith('#'):
                    w = w[1:]
                    g = glove_index_dict.get(w,glove_index_dict.get(w.lower()))

    if g is not None:
        embedding[i,:] = glove_embedding_weights[g,:]
        c+=1

    return embedding


    def generate_word2glove(word2idx):
        word2glove = {}

        for w in word2idx:
            if w in glove_index_dict:
                g = w.lower()
            elif w.lower() in glove_index_dict:
                g = w.lower()
            elif w.startswith('#') and w[1:] in glove_index_dict:
                g = w[1:].lower()
            elif w.startswith('#') and w[1:].lower() in glove_index_dict:
                g = w[1:].lower()
            else:
                continue
    word2glove[w] = g

    return word2glove

    def prepare_data():
        # download the glove file
        filename = maybe_download('http://nlp.stanford.edu/data/',
                                  glove_data_dir + '/glove.6B.zip', 862182613)

        initial_train_data_file = 'data/signalmedia_one_percent.jsonl'

title, content = prepare_training_data(initial_train_data_file, saved_train_data_file)

    glove_embedding_weights = retrieve_glove_embedding_weight(glove_data_dir + '/glove.6B.%dd.txt'%embedding_dim)

# merge title and contents together to retrieve vocabulary
vocab, vocabcount = get_vocab(title + content)

word2idx, idx2word = get_idx(vocab, vocabcount)

own_vocab_embedding = create_own_vocab_embedding(glove_embedding_weights, word2idx)


