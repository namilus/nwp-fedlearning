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


sentences = list(utils.get_sentences("sample_datasets/2_0.txt"))
true_tokens = utils.get_true_tokens(gboard_symbols, sentences)

dataset, input_shape = utils.sentences2dataset(gboard_embedding, gboard_symbols, sentences, batch_size=2)

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

gboard_lstm.build(input_shape=input_shape)
gboard_lstm2 = gboard2.create_gboard_lstm(interpreter)

loss_n = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
loss_a = tf.keras.losses.CategoricalCrossentropy()
loss_sum = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
loss_sumbs = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
# attack.train_model(gboard_lstm2, dataset, 1, optimizer, loss=loss)

true = [[[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],

        [[0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]]]

pred = [[[0.5, 0.25, 0.25],
         [0.25, 0.5, 0.25],
         [0.25, 0.25, 0.5]],

        [[0.25, 0.25, 0.5],
         [0.25, 0.5, 0.25],
         [0.5, 0.25, 0.25]]]

print("loss none")
ln = loss_n(true, pred)
print(ln)
print("loss auto")
print(loss_a(true, pred))
print("loss sum")
print(loss_sum(true, pred))
print("loss sum bs")
print(loss_sumbs(true, pred))

print("correct")
print(tf.reduce_mean(tf.math.reduce_sum(ln, axis=1)))
