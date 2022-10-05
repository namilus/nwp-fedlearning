import time
import random
from functools import reduce
from operator import add
import utils
import metrics
import gboard2

import tensorflow as tf
from tensorflow.random import normal
import numpy as np
from anytree import Node

# the types of `locally' differentially private training techniques
LOCAL_DP_SGD = "DPSGD"
SINGLE_NOISE = "SINGLENOISE"



# need to use reduction NONE because AUTO and SUM_OVER_BATCH_SIZE mess
# up the loss for when you have timesteps ...
CCE = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)


@tf.function(jit_compile=True)
def train_step(x, y, lstm, optimizer, loss=CCE, dp_type=None, dp_stddev=0.0):
    with tf.GradientTape() as tape:
        logits = lstm(x)
        loss_value = loss(y, logits)
    grads = tape.gradient(loss_value, lstm.trainable_weights)
    # add noise
    if dp_type and dp_type == LOCAL_DP_SGD:
        grads = [g + normal(shape=g.shape, mean=0, stddev=dp_stddev) for g in grads]
                
    optimizer.apply_gradients(zip(grads, lstm.trainable_weights))
    return loss_value, grads[-1]

def train_model(lstm, dataset, epochs, optimizer, loss=CCE,
                noise_stddev=0.0, dp_type=LOCAL_DP_SGD,
                after_grad_calc_func=None, after_epoch_func=None):

    """Trains the `lstm' using the passed in parameters. Depending on
    the type of training (LOCAL_DP_SGD or SINGLE_NOISE), we either add
    noise of mean 0 and stddev of `noise_stddev' to every gradient
    (DPSGD) or add noise to the final model's parameters. The
    callables `after_grad_calc_func' and `after_epoch_func' are called
    after every gradient calculation and epoch respectively in order
    to track statistics

    """

    for e in range(epochs):
        start_time = time.time()
        for step, (x_batch_train, y_batch_train) in enumerate(dataset):
            loss_value, grads_fl = train_step(x_batch_train, y_batch_train, lstm, optimizer,
                                              dp_type=dp_type, dp_stddev=noise_stddev)
            if after_grad_calc_func and callable(after_grad_calc_func):
                after_grad_calc_func(e, grads_fl)
            if step % 10 == 0:
                print(f"Loss at step {step}: {tf.reduce_mean(tf.math.reduce_sum(loss_value, axis=1)):.2f}")

        if e == epochs - 1 and dp_type == SINGLE_NOISE:
            # add noise the the final weights of the model
            add_noise_to_model_weights(lstm, stddev=noise_stddev)

        if after_epoch_func and callable(after_epoch_func):
            after_epoch_func(e, lstm)
        print(f"Epoch {e} time : {(time.time() - start_time):.4f}s")

def add_noise_to_model_weights(model, mean=0, stddev=0):
    """ Adds gaussian noise of `mean' and `stddev' to model weights """
    for layer in model.trainable_weights:
        layer.assign_add(normal(layer.shape, mean=0, stddev=stddev))


@tf.function        
def extract_negative_tokens(m1, m2):
    """Subtracts model m2 - m1, and returns the indices in the final
    layer bias of the negative values. If true tokens is provided, it
    uses those to find the best magntidude cutoff to remove the
    noisily flipped words

    """
    m1_final = m1.trainable_weights[-1]
    m2_final = m2.trainable_weights[-1]
    difference = m2_final - m1_final
    return tf.transpose(tf.where(difference < 0)), difference

@tf.function
def extract_best_negative_tokens(m1, m2, true_tokens, stop=gboard2.V):
    m1_final = m1.trainable_weights[-1]
    m2_final = m2.trainable_weights[-1]
    difference = m2_final - m1_final
    # Sort the tokens based on their magnitude
    tokens = tf.expand_dims(tf.cast(tf.argsort(difference), dtype=tf.int64), axis=0)

    # find the best cutoff for f1
    best_f1 = tf.constant(0, dtype=tf.float64)
    cutoff =  tf.constant(0, dtype=tf.int32)

    for i in tf.range(0, limit=stop):
        f1_ = metrics.f1(true_tokens, tokens[:,:i])
        if f1_ > best_f1:
            best_f1 = f1_
            cutoff = i
            
    return tokens[:,:cutoff]


@tf.function
def predict(model, embeddings):
    return model(embeddings)


def next_words(prefix, model, tokens):
    """Returns the next word based on the `prefix'. Only selects from
    tokens present in `tokens' (n,). Returns the next words and their
    normalized probabilities

    """
    # get the last timestep prediction
    nwp = predict(model, prefix)[0][-1].numpy()
    probabilities = [(t, nwp[t]) for t in tokens]
    # renormalize
    total = reduce(add, [p[1] for p in probabilities])
    probabilities = [(t, p/total) for (t, p) in probabilities]
    return probabilities



def scale_model(m1, m2, factor=1):
    """Takes the difference between two models m2 - m1 and applies
    (adds) it to m2 factor times

    """
    difference = []
    for m1_weight, m2_weight in zip(m1.trainable_weights, m2.trainable_weights):
        difference.append(m2_weight - m1_weight)

    for layer, change in zip(m2.trainable_weights, difference) :
        layer.assign_add(change * factor)
    
    
