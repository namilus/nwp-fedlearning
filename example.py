import random
import math
from pprint import pprint
import itertools as it

import utils
import gboard2
import attack
import metrics

import tensorflow as tf

# setup gboard model
interpreter = gboard2.gboard_interpreter("gboard/gboard.tflite")
gboard_lstm = gboard2.create_gboard_lstm(interpreter)
gboard_embedding = gboard2.create_gboard_embedding(interpreter)
gboard_symbols = utils.load_symbols("gboard/gboard.syms")


sentences = ["careful what you type"]

dataset, input_shape = utils.sentences2dataset(gboard_embedding, gboard_symbols, sentences)

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

gboard_lstm.build(input_shape=input_shape)
gboard_lstm2 = gboard2.create_gboard_lstm(interpreter)

attack.train_model(gboard_lstm2, dataset, 1, optimizer)

extracted_tokens, diff = attack.extract_negative_tokens(gboard_lstm2, gboard_lstm)
extracted_tokens = extracted_tokens[0,:].numpy()

for token in extracted_tokens:
    print(token, utils.token2word(gboard_symbols, token), diff[token])

prefixes = it.permutations(extracted_tokens, 1)

generated_sentences = []


for prefix in prefixes:
    prefix = [1] + list(prefix)
    prefix_embeddings = None
    for _ in range(3):
        prefix_embeddings = utils.tokens2embeddings(gboard_embedding, tf.constant([prefix]))
        next_words = attack.next_words(prefix_embeddings, gboard_lstm2, extracted_tokens)
        next_words.sort(key=lambda x: x[1], reverse=True)
        prefix.append(next_words[0][0])

    token_indices = [[0, i, t] for i, t in enumerate(prefix[1:])]
    pp0 = metrics.log_perplexity(prefix_embeddings, token_indices, gboard_lstm)
    pp1 = metrics.log_perplexity(prefix_embeddings, token_indices, gboard_lstm2)
    generated_sentences.append((prefix, ((pp0 - pp1) / pp0).numpy()))


generated_sentences.sort(key=lambda x: x[1], reverse=True)
generated_sentences = generated_sentences[:1]
print(f"generated {len(generated_sentences)} sentences")

for s in generated_sentences:
    print(' '.join([utils.token2word(gboard_symbols, t) for t in s[0]]), s[1])
