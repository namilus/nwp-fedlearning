from attack import loss

import tensorflow as tf
from tensorflow import keras
import argparse as ap
import numpy as np



def dlg(x, y, grad, gradient_func, optimizer=keras.optimizers.Adam(learning_rate=0.01), max_iter=1000):
    """`x', and `y' are initial values of the reconstructed data and
    label. `grad' is the gradient of the training data we're trying to
    reconstruct. `grad_func' is the function that generates gradients.

    """

    x_dummy = tf.Variable(x, name="x_dummy")
    y_dummy = tf.Variable(y, name="y_dummy")


    for i in range(max_iter):
        with tf.GradientTape() as tape:            
            g = gradient_func(x_dummy, y_dummy)
            l = loss(grad, g)
        g2 = tape.gradient(l, [x_dummy, y_dummy])
        if i % 100 == 0:
            print(f"loss @ {i:03d}: {l}")

        optimizer.apply_gradients(zip(g2, [x_dummy, y_dummy]))

    return x_dummy, y_dummy
