import math

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

attack.train_model(gboard_lstm2, dataset, 20, optimizer,
                   # dp_type=attack.LOCAL_DP_SGD, noise_stddev=0.01,
                   # after_epoch_func=after_epoch
                   # after_grad_calc_func=after_g
                   )



def sentence_log_perplexity(embeddings, tokens, model):
    prediction = model.predict(embeddings)
    print("prediction")
    print(prediction)
    probs = tf.gather_nd(prediction, tokens)
    print("probs")
    print(probs)


for s in sentences:
    s = utils.tokenize_sentence(s, gboard_symbols)
    tokens = [utils.word2token(gboard_symbols, w) for w in s]

    tokens_i = [[0, i, t] for i, t in enumerate(tokens[1:])]
    embeddings = utils.tokens2embeddings(gboard_embedding, tf.constant([tokens]))
    print(s, tokens, tokens_i)
    print(embeddings.shape)
    sentence_log_perplexity(embeddings, tokens_i, gboard_lstm2)
    break

