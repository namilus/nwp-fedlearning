import random
import math
from pprint import pprint

import utils
import gboard2
import attack
import metrics
import dlg

import tensorflow as tf
import numpy as np
# setup gboard model
interpreter = gboard2.gboard_interpreter("gboard/gboard.tflite")
gboard_lstm = gboard2.create_gboard_lstm(interpreter)
gboard_embedding = gboard2.create_gboard_embedding(interpreter)
gboard_symbols = utils.load_symbols("gboard/gboard.syms")


sentences = ["how are you doing", "where are you going"]

dataset, input_shape = utils.sentences2dataset(gboard_embedding, gboard_symbols, sentences)

print(input_shape)

x = dataset[0][0]
y = dataset[0][1]

grad = attack.gradient_func(gboard_lstm, x, y)

def g_func(x, y):
    return attack.gradient_func(gboard_lstm, x, y)



def input_label2sentences(input_, label):
    em_t = gboard_embedding.trainable_weights[0]
    data = []
    label_ = []
    for s in range(input_shape[0]):
        sentence = []
        for w in range(input_shape[1]):
            embed = input_[s][w]
            sentence.append(utils.token2word(gboard_symbols, get_closest_token(em_t, embed)))
        data.append(sentence)


    for s in range(input_shape[0]):
        sentence = []
        for w in range(input_shape[1]):
            wlabel = label[s][w]
            sentence.append(utils.token2word(gboard_symbols, np.argsort(wlabel)[-1]))
        label_.append(sentence)

    return data, label_
    

    
def get_closest_token(matrix, embedding):
    v = tf.matmul(tf.expand_dims(embedding, 0), matrix, transpose_b=True)
    v = np.argsort(v.numpy())
    return v[0][-1]
    


print("original")
print(input_label2sentences(x, y))


x_recon, y_recon = dlg.dlg(tf.random.normal(x.shape, mean=0, stddev=0.001),
                           tf.random.normal(y.shape, mean=0, stddev=0.001),
                           grad,
                           g_func,
                           max_iter=1500)

print("dlg reconstructed")
print(input_label2sentences(x_recon, y_recon))
