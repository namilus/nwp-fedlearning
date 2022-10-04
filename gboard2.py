from pathlib import Path
import json
import re


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import cifg

# in the tensor names, the final digits in the name correspond to the
# tensor's index in the interpreter
TENSOR_INDEX_REGEX = r"\d+$"
V = 9502 # vocab size
D = 96  # input embedding dimension
TOTAL_UNITS = 670
UNK_TOKEN = 2
START_TOKEN = 1

def gboard_interpreter(model_file):
    """Returns a tflite interpreter object for the given `model_file'
    Path"""
    if isinstance(model_file, str):
        model_file = Path(model_file)
    with model_file.open('rb') as f:
        b = bytearray(f.read())
        # undo dumb "encryption", see
        # https://hackaday.io/project/164399-android-offline-speech-recognition-natively-on-pc/log/160726-recovering-tflite-models-from-the-binaries
        for i in range(len(b)):
            b[i] ^= 0x1a

    # write this xor'd file to the same directory as the model
    # file
    xor_d = model_file.parent.absolute() / f"{model_file.name}.xor"
    with xor_d.open('wb') as f:
        f.write(b)
    # read the xor'd file and use as the input 
    return tf.lite.Interpreter(model_path=str(xor_d.absolute()), experimental_preserve_all_tensors=True)


def tflite_configuation(config_file=Path("config.json")):
    with config_file.open('r') as f:
        return json.load(f)


def get_tensor_by_name(interpreter, tensor_name):
    """Returns the tensor with `tensor_name' from the tflite
    Interpreter `interpreter'. Tensor names have a number at the end
    of their names, which represent their index in the list of tensors
    returned by interpreter.get_tensor_details()

    """

    # the regex should find the index number at the end of the tensor
    # name.
    
    tensor_index = re.findall(TENSOR_INDEX_REGEX, tensor_name)
    if len(tensor_index) > 1 or not len(tensor_index):
        raise Exception(f"Invalid tensor name {tensor_name}. Tensor names must end in a number that represents the tensor's index")

    tensor_index = int(tensor_index[0])
    tensor_details = list(filter(lambda x: x['index'] == tensor_index, interpreter.get_tensor_details()))

    # tensor details is assumed to be a single element
    return interpreter.get_tensor(tensor_index), tensor_details[0]




def load_tensors(interpreter, tensor_names, quantize=True, transpose=True):
    """Takes the tensors corresponding to those in the
    `tensor_names' argument, unquantize, and concat, and transpose
    them. This function is a thing because the weights for the
    kernel and recurrent kernel are split into three different
    tensors in the tflite model. Also we have to transpose every
    tensor as well so best to put that in a function at least

    """

    tensors = [get_tensor_by_name(interpreter, name) for name in tensor_names]
    if quantize:
        tensors= [tf.cast(t[0], tf.float32) * t[1]["quantization"][0] for t in tensors]

    else:
        tensors = [t[0] for t in tensors] # strip the details 
    if len(tensors) > 1:
        tensors = tf.concat(tensors, axis=0)
    else:
        tensors = tensors[0]

    if transpose:
        return tf.constant(tf.transpose(tensors))
    return tf.constant(tensors)


def init_(tensor):
    return tf.constant_initializer(tensor.numpy())

def create_gboard_lstm(interpreter):
    """ Converts the `interpreter''s gboard tflite model to a Tensorflow model """




    model = keras.Sequential()
    # load the parameter values from the tflite model
    kernel_i = init_(load_tensors(interpreter, ['input2forget_weights4',
                                                'input2cell_weights5',
                                                'input2output_weights6']))
    rkernel_i = init_(load_tensors(interpreter, ['rec2forget_weights8',
                                                 'rec2cell_weights9',
                                                 'rec2output_weights10']))
    bias_i = init_(load_tensors(interpreter, ['forget_gate_bias15',
                                              'cell_gate_bias16',
                                              'output_gate_bias17'], quantize=False))

    proj_kernel_i = init_(load_tensors(interpreter, ['proj_weights18']))
    proj_bias_i = init_(load_tensors(interpreter, ['proj_bias19'], quantize=False))

    fc_kernel_i = init_(load_tensors(interpreter, ['weights23']))
    fc_bias_i = init_(load_tensors(interpreter, ['bias24']))
                         
    # setup TensorFlow model with the kernel and bias initializers
    # pulled from the tflite model above
    model.add(layers.RNN(cifg.CIFGCell(TOTAL_UNITS, D, unit_forget_bias=False,
                                       kernel_initializer=kernel_i,
                                       recurrent_initializer=rkernel_i,
                                       bias_initializer=bias_i,
                                       proj_initializer=proj_kernel_i,
                                       proj_bias_initializer=proj_bias_i),
                         return_sequences=True, unroll=True))

    model.add(layers.Dense(V, kernel_initializer=fc_kernel_i,
                           bias_initializer=fc_bias_i))
    model.add(layers.Softmax())

    return model




def create_gboard_embedding(interpreter):
    """Loads the word embeddings from the gboard `interpreter' and
    creates a tensorflow `Embedding' model """
    embedding_matrix = init_(load_tensors(interpreter, ["weights1"], transpose=False))
    return layers.Embedding(input_dim=V, output_dim=D, mask_zero=True, # </S> is the mask
                            embeddings_initializer=embedding_matrix)
