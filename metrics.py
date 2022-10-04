import tensorflow as tf
from fuzzywuzzy import fuzz


@tf.function
def f1(true_tokens, recovered_tokens):
    """Calculates the f1 score (precision and recall) of the
    attack. `true_tokens' and `recovered_tokens' should be of shapes (1, n)"""
    tp = len(tf.sets.intersection(true_tokens, recovered_tokens).values)
    fp = len(tf.sets.difference(recovered_tokens, true_tokens).values)
    fn = len(tf.sets.difference(true_tokens, recovered_tokens).values)
    try:
        pr = tp / (tp + fp)
        re = tp / (tp + fn)
        f1 = (2 * pr * re) / (pr + re)
        return f1
    except ZeroDivisionError:
        return 0



@tf.function
def log_perplexity(embeddings, tokens, model):
    prediction = model(embeddings)
    probs = tf.gather_nd(prediction, tokens)
    return -1 * tf.math.reduce_sum(tf.map_fn(lambda x: tf.math.log(x), probs))
    




def calc_leven_actual(original_sentences, reconstructed_sentences):
    total = 0
    for r in reconstructed_sentences:
        total += max([leven(r.lower().split(' '), s.lower().split(' ')) for s in original_sentences])
    return total / len(reconstructed_sentences)



def leven(s1, s2):
    """ s1 and s2 are lists of words """
    lookup = {}
    start = 0x21
    def init_lookup(s):
        for word in s:
            if word in lookup: continue
            lookup[word] = chr(start + len(lookup))

    def lookup_word(word):
        return lookup[word]

    init_lookup(s1)
    init_lookup(s2)

    l1 = ' '.join(map(lookup_word, s1))
    l2 = ' '.join(map(lookup_word, s2))

    return fuzz.ratio(l1, l2)
