import random
import math
from pprint import pprint
import itertools as it
from functools import reduce
from operator import add

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


sentences = list(utils.get_sentences("sample_datasets/8_0.txt"))
true_tokens = utils.get_true_tokens(gboard_symbols, sentences)

dataset, input_shape = utils.sentences2dataset(gboard_embedding, gboard_symbols, sentences)

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

gboard_lstm.build(input_shape=input_shape)
gboard_lstm2 = gboard2.create_gboard_lstm(interpreter)

attack.train_model(gboard_lstm2, dataset, 1, optimizer)

extracted_tokens, _ = attack.extract_negative_tokens(gboard_lstm2, gboard_lstm)
extracted_tokens = extracted_tokens[0,:].numpy()


def generate_sentences(nk):
    generated_sentences = []
    prefixes = it.permutations(extracted_tokens, 1)
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
    generated_sentences = generated_sentences[:nk]

    # get the tokens used
    tokens_used = list(set(reduce(add, [s[0] for s in generated_sentences])))

    generated_sentences = [(' '.join([utils.token2word(gboard_symbols, t) for t in s[0][1:]]), s[1]) for s in generated_sentences]
    print(f"\ngenerated {len(generated_sentences)} sentences\n")
    for s in generated_sentences:
        print(s)
        
    leven = metrics.calc_leven_actual(sentences, [s[0] for s in generated_sentences])
    f1_tokens_used = metrics.f1(true_tokens, tf.expand_dims(tf.constant(tokens_used , dtype=tf.int64), axis=0)).numpy()

    print(f"f1 tu {f1_tokens_used}, leven {leven}")
    return generated_sentences




def main():
    print("no scaling")
    generate_sentences(8)

    print("\nscale by 200\n")
    attack.scale_model(gboard_lstm, gboard_lstm2, 200)
    generate_sentences(8)



if __name__ == "__main__":
    main()

