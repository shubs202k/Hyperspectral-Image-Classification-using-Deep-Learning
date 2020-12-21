## Required Imports
import scipy.io as sio
import math
import sklearn.metrics
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.layers import Input, Add, Dense, ReLU, Activation, ZeroPadding3D, Lambda, BatchNormalization 
from tensorflow.keras.layers import Flatten, Conv3D, Conv2D, concatenate, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model


import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
## SG-CNN8/ResNext Model

def initial_convolution_block(samples):
    
    X = Conv2D(64, (3, 3), strides = (2, 2), padding = 'same', name = 'conv_initial',
               input_shape = (20, 20, 64))(samples)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = MaxPooling2D((2, 2), strides = None, name ='MaxPooling_0')(X)

    return X

def group_convolution(y, channels):

    cardinality = 8 #Paths

    assert not channels % cardinality
    
    d = channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    
    groups = []
    
    for j in range(cardinality):
        
        if j % 2 == 0:
            
            no_dilation = Lambda(lambda z: z[:, :, :, j * d:j * d + d])(y)
            groups.append(Conv2D(d, kernel_size=(3, 3), strides = (1,1), padding = 'same')(no_dilation))
        
        else:
            
            dilation_group = Lambda(lambda z: z[:, :, :, j * d:j * d + d])(y)
            x = Conv2D(d, kernel_size = (3, 3), strides = (1,1), padding = 'same', dilation_rate = 1)(dilation_group)
            x = Conv2D(d, kernel_size = (3, 3), strides = (1,1), padding = 'same', dilation_rate = 3)(x)
            x = Conv2D(d, kernel_size = (3, 3), strides = (1,1), padding = 'same', dilation_rate = 5)(x)
            groups.append(x)

            
    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = concatenate(groups)

    return y

def channel_shuffle(x):

    cardinality = 8
    
    b, h, w, c = x.shape
    
    x = tf.reshape(x, [-1, h, w, cardinality, c // cardinality])
    x = tf.transpose(x, perm = [0, 1, 2, 4, 3])
    x = tf.reverse(x,[-1])
    x = tf.reshape(x, [-1, h, w, c])
    
    return x

def SG_Unit(X):
    
    # Save the input value
    X_shortcut = X
    l2_ = 0.01
    X = Conv2D(64, (1, 1), kernel_regularizer = regularizers.l2(l2_), padding = "same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Group convolution
    X = group_convolution(X, 64)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D(128, (1, 1), kernel_regularizer = regularizers.l2(l2_), padding = "same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # channel shuffle
    X = channel_shuffle(X)
    print('After shuffle :',X.shape)
    
    X_shortcut = Conv2D(128, (1, 1), kernel_regularizer = regularizers.l2(l2_), padding = "same")(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)
    X_shortcut = Activation('relu')(X_shortcut)


    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def model(input_shape, classes):
    
    X_input = Input(input_shape)
    X = initial_convolution_block(X_input)
    X = SG_Unit(X) # shuffled group convolution unit
    X = GlobalAveragePooling2D()(X)
    X = Dense(256, input_dim = X.shape, activation = 'relu', name = 'final_fully_connected', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = Dense(classes, input_dim = X.shape, activation = 'softmax')(X)
    model = Model(inputs = X_input, outputs = X)
    
    return model

def fine_tune_target(input_shape, classes):
    X_input = Input(input_shape)

    X = Dense(256, input_dim = X_input.shape, activation = 'relu', name = 'fc_256',
              kernel_initializer = glorot_uniform(seed = 0))(X_input)

    X = Dense(classes, input_dim = X.shape, activation = 'softmax', name = 'fc' + str(classes),
              kernel_initializer = glorot_uniform(seed = 0))(X)

    model = Model(inputs = X_input, outputs = X, name = "fine_tune")

    return model
