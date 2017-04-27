import keras.backend as K

from keras.layers import Lambda

POWER = 2

def softmaxLayer(x):
    channel_sum = K.sum(x,axis=3)
    softmax = K.expand_dims(K.softmax(channel_sum),axis=-1)
    return softmax

def softmaxLayer_output_shape(input_shape):
    return input_shape

def sumAbsVal(x):
    abs_channel_sum = K.sum(K.abs(x),axis=2)
    return abs_channel_sum

def sumAbsVal_output_shape(input_shape):
    return input_shape

def maxoutLayer(x):
    channel_max = K.max(x,axis=2)
    attention = K.pow(channel_max,POWER)
    return attention

def maxoutLayer_output_shape(input_shape):
    return input_shape