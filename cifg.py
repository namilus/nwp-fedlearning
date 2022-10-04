import tensorflow as tf
from tensorflow import keras
from keras.layers import AbstractRNNCell

class CIFGCell(AbstractRNNCell):
    def __init__(self,
                 units,
                 proj_units,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 proj_initializer='glorot_uniform',
                 proj_bias_initializer='zeros',
                 proj_activation='linear',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        self.units = units
        self.proj_units = proj_units
        super(CIFGCell, self).__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.recurrent_activation = keras.activations.get(recurrent_activation)
        self.proj_activation = keras.activations.get(proj_activation)
        self.use_bias = use_bias
        # holds weights for the forget, cell, and output gates
        # associated with the input, they are concatenated in the order
        # W_fx, W_cx, W_ox
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        # holds weights for the forget, cell, and output gates
        # associated with the recburrent output, the are concatenated in the order
        # W_fh, W_ch, W_oh
        self.recurrent_initializer = keras.initializers.get(recurrent_initializer)

        # Extra part of this CIFG, shrinks the h_t output to be of proj_units
        # dimensionality
        self.proj_initializer = keras.initializers.get(proj_initializer)

        # holds the bias for the 3 gates in the order forget, cell, output
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # bias for the proj weights
        self.proj_bias_initializer = keras.initializers.get(proj_bias_initializer)
        self.unit_forget_bias = unit_forget_bias
        

        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = keras.constraints.get(recurrent_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        # implementation = kwargs.pop('implementation', 1)
        # if self.recurrent_dropout != 0 and implementation != 1:
        #   logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
        self.implementation = 0
        # else:
        #   self.implementation = implementation
        # self.state_size = [self.units, self.units]
        
        # self.output_size = self.units



    @property
    def state_size(self):
        return [self.proj_units, self.units]

    @property
    def output_size(self):
        return self.proj_units

    
    def build(self, input_shape):
        input_dim = input_shape[-1] # 96
        self.kernel = self.add_weight(shape=(input_dim, self.units * 3), # self.units * 3
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      name='kernel')

        self.recurrent_kernel = self.add_weight(shape=(self.proj_units, self.units * 3),
                                                initializer=self.recurrent_initializer,
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint,
                                                name='recurrent_kernel')

        self.proj_kernel = self.add_weight(shape=(self.units, self.proj_units),
                                           initializer=self.proj_initializer,
                                           name="proj_kernel")

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return keras.backend.concatenate([
                        keras.initializers.get('ones')((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])

            else:
                bias_initializer = self.bias_initializer

            proj_bias_initializer = self.proj_bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 3,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)

            self.proj_bias = self.add_weight(
                shape=(self.proj_units,),
                trainable=False, # because loaded as zero, makes think its not trainable
                name="proj_bias",
                initializer=proj_bias_initializer)
        else:
            self.bias = None
            self.proj_bias = None
        self.built = True



    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using fused kernels."""
        x_f, x_c, x_o = x
        h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        kr_f, kr_c, kr_o = tf.split(self.recurrent_kernel, num_or_size_splits=3, axis=1)
        f = self.recurrent_activation(x_f + keras.backend.dot(h_tm1_f, kr_f))
        i = tf.math.subtract(float(1), f)
        c = f * c_tm1 + i * self.activation(x_c + keras.backend.dot(h_tm1_c, kr_c))
        o = self.recurrent_activation(x_o + keras.backend.dot(h_tm1_o, kr_o))

        return c, o
            
    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs

        k_f, k_c, k_o = tf.split(self.kernel, num_or_size_splits=3, axis=1)

        x_f = keras.backend.dot(inputs_f, k_f)
        x_c = keras.backend.dot(inputs_c, k_c)
        x_o = keras.backend.dot(inputs_o, k_o)

        if self.use_bias:
            b_f, b_c, b_o = tf.split(self.bias, num_or_size_splits=3, axis=0)
            x_f = keras.backend.bias_add(x_f, b_f)
            x_c = keras.backend.bias_add(x_c, b_c)
            x_o = keras.backend.bias_add(x_o, b_o)

        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1

        # x = (x_f, x_c, x_o)
        # h_tm1 = (h_tm1_f, h_tm1_c, h_tm1_o)
        # c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
        kr_f, kr_c, kr_o = tf.split(self.recurrent_kernel, num_or_size_splits=3, axis=1)
        f = self.recurrent_activation(x_f + keras.backend.dot(h_tm1, kr_f))
        i = tf.math.subtract(1., f)
        c = f * c_tm1 + i * self.activation(x_c + keras.backend.dot(h_tm1, kr_c))
        o = self.recurrent_activation(x_o + keras.backend.dot(h_tm1, kr_o))

        h = o * self.activation(c)

        # projection stuff now
        w_p = self.proj_kernel
        h_p = keras.backend.dot(h, w_p)

        if self.use_bias:
            h_p = keras.backend.bias_add(h_p, self.proj_bias)

        h_p = self.proj_activation(h_p)
        
        return h_p, [h_p, c]                  
