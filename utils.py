import difflib
import re
from pathlib import Path
from operator import add
from functools import reduce

import gboard2
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences



def tokenize_sentence(sentence, symbols, UNK="<UNK>", S="<S>", add_start_token=True, fix_spelling=True):
    """Tokenizes sentence, removes non alphanumeric, adds the unknown
    token for words in `sentence' not included in `symbols' (list of words)"""
    sentence = re.sub(r'[!"#$%&''()*+,-./:;<=>?@\[\]^_`{|}~]', ' ',sentence)
    sentence = sentence.lower()
    words = sentence.split()
    if add_start_token:
        sentence=[S] # add start of sentence token
    else:
        sentence = []
    for w in words:
        if w not in symbols:
            if w=="i":
                w="I"
            if fix_spelling:
                matches = difflib.get_close_matches(w, symbols, n=1, cutoff=0.8)
                if matches:
                    # try to fix spelling mistake
                    sentence.append(matches[0])
                else:
                    sentence.append(UNK)
            else:
                sentence.append(UNK)
        else:
            sentence.append(w)
    return sentence

def word2token(symbol_token_dict, word):
    return symbol_token_dict[word]

def token2word(symbol_token_dict, token):
    vals = list(symbol_token_dict.values())
    posn = vals.index(token)
    return list(symbol_token_dict.keys())[posn]

def tokens2embeddings(embedding_m, tokens):
    """Converts a tensor of `tokens' to a list of embeddings"""
    return embedding_m(tokens)


def load_symbols(symbol_file):
    """Loads the symbols to token map from the `symbol_file' Path
    argument

    """
    symbols = {}
    if isinstance(symbol_file, str):
        symbol_file = Path(symbol_file)
    with symbol_file.open('r') as f:
        for l in f:
            symbol, token = tuple(l.strip().split('\t'))
            symbols[symbol] = int(token)
    return symbols


def batch_slices(dataset_size, batch_size):
    slices = []
    if dataset_size % batch_size != 0:
        raise Exception("Invalid batch size. Batch size must be divisible by the dataset length")
    for i in range(int(dataset_size / batch_size)):
        start = i * batch_size
        end = start + batch_size
        if end > dataset_size:
            end = dataset_size
        yield slice(start, end)

def sentences2dataset(embedding_m, symbols_token_dict, sentences, batch_size=None):
    """Convert an array of sentences (strings) into training samples
    e.g data and label for the gboard lstm """

    symbols = symbols_token_dict.keys()
    # tokenize, convert to tokens, and pad
    sentences = [tokenize_sentence(s, symbols) for s in sentences]
    sentences = [[word2token(symbols_token_dict, w) for w in s] for s in sentences]
    sentences = pad_sequences(sentences, padding='post')
    input_ = tokens2embeddings(embedding_m, sentences[:,:-1])
    label  = tf.one_hot(sentences[:,1:], gboard2.V, dtype=tf.float32)

    dataset = []
    if batch_size:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise Exception("Invalid batch size. Must be an integer greater than zero")
        for slice_ in batch_slices(len(sentences), batch_size):
            dataset.append((input_[slice_,:], label[slice_,:]))

    else:
        dataset = [(input_, label)] # full batch

    input_shape = dataset[0][0].shape
            
    return dataset, input_shape



def get_sentences(dataset_file):
    """Reads the sentences in `dataset_file', which is assumed to have
    a new sentence on each newline. Yields each new sentence.

    """
    if isinstance(dataset_file, str):
        dataset_file = Path(dataset_file)
    with dataset_file.open('r') as f:
        for l in f:
            yield l.strip()
    


def get_true_tokens(symbols_tokens_dict, sentences):
    symbols = symbols_tokens_dict.keys()
    sentences = [tokenize_sentence(s, symbols, add_start_token=False, fix_spelling=False) for s in sentences]
    sentences = [[word2token(symbols_tokens_dict, w) for w in s] for s in sentences]
    sentences = tf.constant(reduce(add, sentences))
    unique_tokens, _ = tf.unique(sentences)
    
    return tf.expand_dims(tf.cast(unique_tokens, dtype=tf.int64), axis=0)


def output_results(file_, results):
    """Takes a list of tuples and writes them to `file_', line by
    line. File must be an str or a Path object

    """
    if isinstance(file_, str):
        file_ = Path(file_)
    with file_.open('w') as f:
        for record in results:
            print(*record, file=f)
