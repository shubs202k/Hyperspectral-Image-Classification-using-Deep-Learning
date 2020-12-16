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

## Pick samples belonging to all classes

def pick_samples_from_class(Class, cube_size, data, ground_truth, cubes, output_class, overlap_ratio, channels):
    
    ## Get row and column position from ground truth image for class
    class_indices = np.where(ground_truth == Class)
    
    ## Remove border position class samples
    class_cube_positions = [[class_indices[0][i], class_indices[1][i]] for i in range(len(class_indices[0])) 
                        if len(ground_truth) - np.ceil(cube_size / 2) > class_indices[0][i] > np.ceil(cube_size / 2) 
                        and len(ground_truth[0]) - np.ceil(cube_size / 2) > class_indices[1][i] > np.ceil(cube_size / 2)]
    
    #print('Length of class positions', len(class_cube_positions))
    
    extracted_cubes = [[class_cube_positions[0][0], class_cube_positions[0][1]]]
    
    ## Form the first cube for this class
    cubes.append(np.array(data[class_cube_positions[0][0] - int(cube_size / 2):class_cube_positions[0][0] + int(cube_size / 2),
                       (class_cube_positions[0][1] - int(cube_size / 2)):class_cube_positions[0][1] + int(cube_size / 2),
                         :channels]))
    
    ## Output class value
    output_class.append(Class)
        
    ## Pick cube/sample if it satisfies the criteria for the overlap ratio
    for i in range(1, len(class_cube_positions)):
        
        distance_vector = [] ## Calculate distance from existing sample to the next candiddate cube sample
        
        for k in range(len(extracted_cubes)):
            
            distance = math.sqrt((class_cube_positions[i][0] - extracted_cubes[k][0]) ** 2 + 
                                 (class_cube_positions[i][1] - extracted_cubes[k][1]) ** 2)
            
            distance_vector.append(distance)
            
        if np.min(distance_vector) > int(cube_size * (1 - overlap_ratio)):
            
            cubes.append(np.array(data[class_cube_positions[i][0] - int(cube_size / 2):class_cube_positions[i][0] + int(cube_size / 2),
                                      (class_cube_positions[i][1] - int(cube_size / 2)):class_cube_positions[i][1] + int(cube_size / 2),
                                      :channels]))
            
            output_class.append(Class)
            extracted_cubes.append([class_cube_positions[i][0], class_cube_positions[i][1]])
            
    return cubes, output_class, extracted_cubes

## Collect and combine samples from all classes

def collect_samples_from_all_classes(classes, cube_size, data, ground_truth, cubes, output_class, overlap_ratio, channels):
    
    class_samples = []
    
    for Class in classes:
        cubes, output_class, extracted_cubes = pick_samples_from_class(Class, cube_size, data, ground_truth, cubes, 
                                                                       output_class,overlap_ratio, channels)
        class_samples.append(len(extracted_cubes))
    
    cubes = np.array(cubes)
    output_class = np.array(output_class)
    
    print('Class Samples : ', class_samples)
    
    return cubes, output_class, class_samples

## Prepare Training, Validation & Test Data

def get_training_and_test_set_for_indian_pines(training_samples_from_each_class,
                                               test_samples_from_each_class,
                                               class_samples, cubes, output_class):
    
    class_2_samples = cubes[np.where(output_class == 2)[0]]
    class_2_labels = output_class[np.where(output_class == 2)[0]]

    class_3_samples = cubes[np.where(output_class == 3)[0]]
    class_3_labels = output_class[np.where(output_class == 3)[0]]

    class_5_samples = cubes[np.where(output_class == 5)[0]]
    class_5_labels = output_class[np.where(output_class == 5)[0]]

    class_6_samples = cubes[np.where(output_class == 6)[0]]
    class_6_labels = output_class[np.where(output_class == 6)[0]]

    class_8_samples = cubes[np.where(output_class == 8)[0]]
    class_8_labels = output_class[np.where(output_class == 8)[0]]

    class_10_samples = cubes[np.where(output_class == 10)[0]]
    class_10_labels = output_class[np.where(output_class == 10)[0]]
    
    class_11_samples = cubes[np.where(output_class == 11)[0]]
    class_11_labels = output_class[np.where(output_class == 11)[0]]
    
    class_12_samples = cubes[np.where(output_class == 12)[0]]
    class_12_labels = output_class[np.where(output_class == 12)[0]]
    
    class_14_samples = cubes[np.where(output_class == 14)[0]]
    class_14_labels = output_class[np.where(output_class == 14)[0]]


    class_samples_collection = [class_2_samples, class_3_samples, class_5_samples, class_6_samples,
                               class_8_samples, class_10_samples, class_11_samples, class_12_samples, class_14_samples]

    class_labels_collection = [class_2_labels, class_3_labels, class_5_labels, class_6_labels,
                               class_8_labels, class_10_labels, class_11_labels, class_12_labels, class_14_labels]

    # Training & Test Set Arrays
    X_train = []
    X_test = []

    y_train = []
    y_test = []

    # Get Training set size samples from each class
    for samples in class_samples_collection:
        
        X_train.append(samples[0:training_samples_from_each_class])
        
        X_test.append(samples[training_samples_from_each_class:
                              training_samples_from_each_class +  
                              test_samples_from_each_class])
        
    # Get output labels
    for labels in class_labels_collection:
        y_train.append(labels[0:training_samples_from_each_class])
        
        y_test.append(labels[training_samples_from_each_class:
                             training_samples_from_each_class +  
                              test_samples_from_each_class])

    X_train = np.concatenate(X_train, axis = 0)
    X_test = np.concatenate(X_test, axis = 0)

    y_train = np.concatenate(y_train, axis = 0)
    y_test = np.concatenate(y_test, axis = 0)

    
    ## Shuffle Training Set
    samples_train = np.arange(X_train.shape[0])
    np.random.shuffle(samples_train)

    X_train = X_train[samples_train]
    y_train = y_train[samples_train]


    ## Shuffle Test Set
    samples_test = np.arange(X_test.shape[0])
    np.random.shuffle(samples_test)

    X_test = X_test[samples_test]
    y_test = y_test[samples_test]

    # Get counts(samples) of each class in test set
    values_test_set, counts_test_set = np.unique(y_test, return_counts = True)
    values_training_set, counts_training_set = np.unique(y_train, return_counts = True)


    print("Samples per class: " + str(class_samples) + '\n'
          "Total number of samples is " + str(np.sum(class_samples)) + '.\n')
    
    print("unique classes in training set: " + str(values_training_set) + '\n'
          "Total number of samples in training set is " + str(np.sum(counts_training_set)) + '.\n'
          "Samples per class in training set: " + str(counts_training_set) + '\n')

    print("unique classes in test set: " + str(values_test_set) + '\n'
          "Total number of samples in test set is " + str(np.sum(counts_test_set)) + '.\n'
          "Samples per class in test set: " + str(counts_test_set) + '\n')
    print('\n')

    ## one hot encode labels
    onehot_encoder = OneHotEncoder(sparse = False)

    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)

    y_train = onehot_encoder.fit_transform(y_train)
    y_test = onehot_encoder.fit_transform(y_test)

    return X_train, X_test, y_train, y_test, counts_test_set, class_samples

def get_training_and_test_set_for_pavia(training_samples_from_each_class,
                                        test_samples_from_each_class,
                                        class_samples, cubes, output_class):
    
    class_1_samples = cubes[np.where(output_class == 1)[0]]
    class_1_labels = output_class[np.where(output_class == 1)[0]]

    class_2_samples = cubes[np.where(output_class == 2)[0]]
    class_2_labels = output_class[np.where(output_class == 2)[0]]

    class_3_samples = cubes[np.where(output_class == 3)[0]]
    class_3_labels = output_class[np.where(output_class == 3)[0]]

    class_4_samples = cubes[np.where(output_class == 4)[0]]
    class_4_labels = output_class[np.where(output_class == 4)[0]]

    class_5_samples = cubes[np.where(output_class == 5)[0]]
    class_5_labels = output_class[np.where(output_class == 5)[0]]

    class_6_samples = cubes[np.where(output_class == 6)[0]]
    class_6_labels = output_class[np.where(output_class == 6)[0]]

    class_7_samples = cubes[np.where(output_class == 7)[0]]
    class_7_labels = output_class[np.where(output_class == 7)[0]]

    class_8_samples = cubes[np.where(output_class == 8)[0]]
    class_8_labels = output_class[np.where(output_class == 8)[0]]

    class_9_samples = cubes[np.where(output_class == 9)[0]]
    class_9_labels = output_class[np.where(output_class == 9)[0]]


    class_samples_collection = [class_1_samples, class_2_samples, class_3_samples, class_4_samples, class_5_samples,
                               class_6_samples, class_7_samples, class_8_samples, class_9_samples]

    class_labels_collection = [class_1_labels, class_2_labels, class_3_labels, class_4_labels, class_5_labels,
                              class_6_labels, class_7_labels, class_8_labels, class_9_labels]

    # Training & Test Set Arrays
    X_train = []
    X_test = []

    y_train = []
    y_test = []

    # Get Training set size samples from each class
    for samples in class_samples_collection:
        
        X_train.append(samples[0:training_samples_from_each_class])
        
        X_test.append(samples[training_samples_from_each_class:
                              training_samples_from_each_class +  
                              test_samples_from_each_class])
        
    # Get output labels
    for labels in class_labels_collection:
        y_train.append(labels[0:training_samples_from_each_class])
        
        y_test.append(labels[training_samples_from_each_class:
                             training_samples_from_each_class +  
                             test_samples_from_each_class])

    X_train = np.concatenate(X_train, axis = 0)
    X_test = np.concatenate(X_test, axis = 0)

    y_train = np.concatenate(y_train, axis = 0)
    y_test = np.concatenate(y_test, axis = 0)

    ## Shuffle Training Set
    samples_train = np.arange(X_train.shape[0])
    np.random.shuffle(samples_train)

    X_train = X_train[samples_train]
    y_train = y_train[samples_train]

    ## Shuffle Test Set
    samples_test = np.arange(X_test.shape[0])
    np.random.shuffle(samples_test)

    X_test = X_test[samples_test]
    y_test = y_test[samples_test]

    # Get counts(samples) of each class in test set
    values_test_set, counts_test_set = np.unique(y_test, return_counts = True)
    values_training_set, counts_training_set = np.unique(y_train, return_counts = True)


    print("Samples per class: " + str(class_samples) + '\n'
          "Total number of samples is " + str(np.sum(class_samples)) + '.\n')
    
    print("unique classes in training set: " + str(values_training_set) + '\n'
          "Total number of samples in training set is " + str(np.sum(counts_training_set)) + '.\n'
          "Samples per class in training set: " + str(counts_training_set) + '\n')

    print("unique classes in test set: " + str(values_test_set) + '\n'
          "Total number of samples in test set is " + str(np.sum(counts_test_set)) + '.\n'
          "Samples per class in test set: " + str(counts_test_set) + '\n')
    print('\n')

    ## one hot encode labels
    onehot_encoder = OneHotEncoder(sparse = False)

    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)

    y_train = onehot_encoder.fit_transform(y_train)
    y_test = onehot_encoder.fit_transform(y_test)

    return X_train, X_test, y_train, y_test, counts_test_set, class_samples

def get_training_and_test_set_for_botswana(training_samples_from_each_class,
                                           test_samples_from_each_class,
                                          class_samples, cubes, output_class):

    class_1_samples = cubes[np.where(output_class == 1)[0]]
    class_1_labels = output_class[np.where(output_class == 1)[0]]
    
    class_2_samples = cubes[np.where(output_class == 2)[0]]
    class_2_labels = output_class[np.where(output_class == 2)[0]]

    class_3_samples = cubes[np.where(output_class == 3)[0]]
    class_3_labels = output_class[np.where(output_class == 3)[0]]

    class_4_samples = cubes[np.where(output_class == 4)[0]]
    class_4_labels = output_class[np.where(output_class == 4)[0]]

    class_5_samples = cubes[np.where(output_class == 5)[0]]
    class_5_labels = output_class[np.where(output_class == 5)[0]]

    class_6_samples = cubes[np.where(output_class == 6)[0]]
    class_6_labels = output_class[np.where(output_class == 6)[0]]

    class_7_samples = cubes[np.where(output_class == 7)[0]]
    class_7_labels = output_class[np.where(output_class == 7)[0]]

    class_8_samples = cubes[np.where(output_class == 8)[0]]
    class_8_labels = output_class[np.where(output_class == 8)[0]]

    class_9_samples = cubes[np.where(output_class == 9)[0]]
    class_9_labels = output_class[np.where(output_class == 9)[0]]

    class_10_samples = cubes[np.where(output_class == 10)[0]]
    class_10_labels = output_class[np.where(output_class == 10)[0]]
    
    class_11_samples = cubes[np.where(output_class == 11)[0]]
    class_11_labels = output_class[np.where(output_class == 11)[0]]
    
    class_12_samples = cubes[np.where(output_class == 12)[0]]
    class_12_labels = output_class[np.where(output_class == 12)[0]]

    class_13_samples = cubes[np.where(output_class == 13)[0]]
    class_13_labels = output_class[np.where(output_class == 13)[0]]
    
    class_14_samples = cubes[np.where(output_class == 14)[0]]
    class_14_labels = output_class[np.where(output_class == 14)[0]]


    class_samples_collection = [class_1_samples, class_2_samples, class_3_samples, class_4_samples, class_5_samples, class_6_samples,
                               class_7_samples, class_8_samples, class_9_samples, class_10_samples, class_11_samples, class_12_samples,
                               class_13_samples, class_14_samples]

    class_labels_collection = [class_1_labels, class_2_labels, class_3_labels, class_4_labels, class_5_labels, class_6_labels,
                               class_7_labels, class_8_labels, class_9_labels, class_10_labels, class_11_labels, class_12_labels,
                               class_13_labels, class_14_labels]

    # Training & Test Set Arrays
    X_train = []
    X_test = []

    y_train = []
    y_test = []

    # Get Training set size samples from each class
    for samples in class_samples_collection:
        
        X_train.append(samples[0:training_samples_from_each_class])
        
        X_test.append(samples[training_samples_from_each_class:
                              training_samples_from_each_class +  
                              test_samples_from_each_class])
        
    # Get output labels
    for labels in class_labels_collection:
        y_train.append(labels[0:training_samples_from_each_class])
        
        y_test.append(labels[training_samples_from_each_class:
                             training_samples_from_each_class +  
                             test_samples_from_each_class])

    X_train = np.concatenate(X_train, axis = 0)
    X_test = np.concatenate(X_test, axis = 0)

    y_train = np.concatenate(y_train, axis = 0)
    y_test = np.concatenate(y_test, axis = 0)

    
    ## Shuffle Training Set
    samples_train = np.arange(X_train.shape[0])
    np.random.shuffle(samples_train)

    X_train = X_train[samples_train]
    y_train = y_train[samples_train]


    ## Shuffle Test Set
    samples_test = np.arange(X_test.shape[0])
    np.random.shuffle(samples_test)

    X_test = X_test[samples_test]
    y_test = y_test[samples_test]

    # Get counts(samples) of each class in test set
    values_test_set, counts_test_set = np.unique(y_test, return_counts = True)
    values_training_set, counts_training_set = np.unique(y_train, return_counts = True)


    print("Samples per class: " + str(class_samples) + '\n'
          "Total number of samples is " + str(np.sum(class_samples)) + '.\n')
    
    print("unique classes in training set: " + str(values_training_set) + '\n'
          "Total number of samples in training set is " + str(np.sum(counts_training_set)) + '.\n'
          "Samples per class in training set: " + str(counts_training_set) + '\n')

    print("unique classes in test set: " + str(values_test_set) + '\n'
          "Total number of samples in test set is " + str(np.sum(counts_test_set)) + '.\n'
          "Samples per class in test set: " + str(counts_test_set) + '\n')
    print('\n')

    ## one hot encode labels
    onehot_encoder = OneHotEncoder(sparse = False)

    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)

    y_train = onehot_encoder.fit_transform(y_train)
    y_test = onehot_encoder.fit_transform(y_test)

    return X_train, X_test, y_train, y_test, counts_test_set, class_samples

def sample_extraction_from_indian_pines(classes,
                                        cube_size,
                                        data,
                                        ground_truth,
                                        cubes,
                                        output_class,
                                        training_samples_from_each_class,
                                        test_samples_from_each_class,
                                        overlap_ratio,
                                        channels):
    
    cubes, output_class, class_samples = collect_samples_from_all_classes(classes, 
                                                                          cube_size, 
                                                                          data,  
                                                                          ground_truth, 
                                                                          cubes, 
                                                                          output_class , 
                                                                          overlap_ratio, 
                                                                          channels)
    
    X_train, X_test, y_train, y_test, counts_test_set, class_samples = get_training_and_test_set_for_indian_pines(
                                                                            training_samples_from_each_class,
                                                                            test_samples_from_each_class,
                                                                            class_samples, 
                                                                            cubes,
                                                                            output_class)
    
    return X_train, X_test, y_train, y_test, counts_test_set, class_samples

def sample_extraction_from_pavia(classes,
                                 cube_size,
                                 data,
                                 ground_truth,
                                 cubes,
                                 output_class,
                                 training_samples_from_each_class,
                                 test_samples_from_each_class,
                                 overlap_ratio,
                                 channels):
    
    cubes, output_class, class_samples = collect_samples_from_all_classes(classes, 
                                                                          cube_size, 
                                                                          data,  
                                                                          ground_truth, 
                                                                          cubes, 
                                                                          output_class , 
                                                                          overlap_ratio, 
                                                                          channels)
    
    X_train, X_test, y_train, y_test, counts_test_set, class_samples = get_training_and_test_set_for_pavia(
                                                                            training_samples_from_each_class,
                                                                            test_samples_from_each_class,
                                                                            class_samples, 
                                                                            cubes,
                                                                            output_class)
    
    return X_train, X_test, y_train, y_test, counts_test_set, class_samples

def sample_extraction_from_botswana(classes,
                                   cube_size,
                                   data,
                                   ground_truth,
                                   cubes,
                                   output_class,
                                   training_samples_from_each_class,
                                   test_samples_from_each_class,
                                   overlap_ratio,             
                                   channels):
    
    cubes, output_class, class_samples = collect_samples_from_all_classes(classes, 
                                                                          cube_size, 
                                                                          data,  
                                                                          ground_truth, 
                                                                          cubes, 
                                                                          output_class , 
                                                                          overlap_ratio, 
                                                                          channels)
    
    X_train, X_test, y_train, y_test, counts_test_set, class_samples = get_training_and_test_set_for_botswana(
                                                                            training_samples_from_each_class,
                                                                            test_samples_from_each_class,
                                                                            class_samples, 
                                                                            cubes,
                                                                            output_class)
    
    return X_train, X_test, y_train, y_test, counts_test_set, class_samples



## SG-CNN8/ResNext Model

def initial_convolution_block(samples):
    
    X = Conv2D(64, (3, 3), strides = (2, 2), padding = 'same', name = 'conv_initial',
               input_shape = (20, 20, 64))(samples)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = MaxPooling2D((2, 2), strides = None, name='MaxPooling_0')(X)

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
            x = Conv2D(d, kernel_size=(3, 3), strides = (1,1), padding='same', dilation_rate = 1)(dilation_group)
            x = Conv2D(d, kernel_size=(3, 3), strides = (1,1), padding='same', dilation_rate = 3)(x)
            x = Conv2D(d, kernel_size=(3, 3), strides = (1,1), padding='same', dilation_rate = 5)(x)
            groups.append(x)

            
    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = concatenate(groups)

    return y

def SG_Unit(X):
    
    # Save the input value
    X_shortcut = X
    l2_ = 0.01
    X = Conv2D(64, (1, 1), kernel_regularizer = regularizers.l2(l2_), padding = "same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Group convolution
    X = group_convolution(X, 64)
    print('Groud convolution output ', X.shape)
    print(X[3])
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    # Channel shuffle
    #shuffle_channels = np.arange(X.shape[2])
    #np.random.shuffle(shuffle_channels)
    #X = X[:,:,shuffle_channels]
    
    X = Conv2D(128, (1, 1), kernel_regularizer = regularizers.l2(l2_), padding = "same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X_shortcut = Conv2D(128, (1, 1), kernel_regularizer = regularizers.l2(l2_), padding = "same")(X_shortcut)
    X_shortcut = BatchNormalization()(X_shortcut)
    X_shortcut = Activation('relu')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def model(input_shape, classes):
    
    X_input = Input(input_shape)
    X = initial_convolution_block(X_input)
    X = SG_Unit(X)
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

def pretrain_source_models(training_set_size,
                           test_samples_from_each_class,
                           classes,
                           cube_size,
                           overlap_ratio,
                           data,
                           ground_truth,
                           batch_size,
                           channels,
                           epochs,
                           Verbosity,
                           accuracies,
                           learning_rate,
                           source):
    
    for i in range(len(training_set_size)):
        
        print("\n=============================================================================================================\n"
              "Model training starts for data with " + str(int(training_set_size[i])) + " samples from each class in training set\n"
              "==============================================================================================================\n")



        if source == 'indian_pines':
            
            X_train, X_test, y_train, y_test, counts_test_set, class_samples = sample_extraction_from_indian_pines(classes = classes, 
                                                                                                                    cube_size = cube_size, 
                                                                                                                    data = data, 
                                                                                                                    ground_truth = ground_truth, 
                                                                                                                    cubes = [], 
                                                                                                                    output_class = [], 
                                                                                                                    training_samples_from_each_class = training_set_size[i],
                                                                                                                    test_samples_from_each_class = test_samples_from_each_class,
                                                                                                                    overlap_ratio = overlap_ratio, 
                                                                                                                    channels = channels)
        elif source == 'pavia':

            X_train, X_test, y_train, y_test, counts_test_set, class_samples = sample_extraction_from_pavia(classes = classes, 
                                                                                                           cube_size = cube_size, 
                                                                                                           data = data, 
                                                                                                           ground_truth = ground_truth, 
                                                                                                           cubes = [], 
                                                                                                           output_class = [], 
                                                                                                           training_samples_from_each_class = training_set_size[i],
                                                                                                           test_samples_from_each_class = test_samples_from_each_class, 
                                                                                                           overlap_ratio = overlap_ratio, 
                                                                                                           channels = channels)
        elif source == 'botswana':

            X_train, X_test, y_train, y_test, counts_test_set, class_samples = sample_extraction_from_botswana(classes = classes, 
                                                                                                               cube_size = cube_size, 
                                                                                                               data = data, 
                                                                                                               ground_truth = ground_truth, 
                                                                                                               cubes = [], 
                                                                                                               output_class = [], 
                                                                                                               training_samples_from_each_class = training_set_size[i],
                                                                                                               test_samples_from_each_class = test_samples_from_each_class,
                                                                                                               overlap_ratio = overlap_ratio, 
                                                                                                               channels = channels)
            
        print('X_train => ' + str(X_train.shape) + '\n' +
              'X_test  => ' + str(X_test.shape) + '\n' +
              'y_train => ' + str(y_train.shape) + '\n' +
              'y_test  => ' + str(y_test.shape) + '\n')

        X_train = np.array(X_train).astype(np.float32)
        X_test = np.array(X_test).astype(np.float32)

        model_to_train = model(input_shape = X_train[0].shape, classes = len(classes))
        model_to_train.summary()

        # save best model
        model_checkpoint = ModelCheckpoint('/content/drive/My Drive/Hyperspectral_Image_Classification/code//Trained_models//full_models//' + source + '_as_source_with_' 
                                           + str(int(training_set_size[i])) 
                                           + ' samples_from_each_class_in_training_set.h5',
                                            monitor = 'val_categorical_accuracy', 
                                            verbose = 1, 
                                            save_best_only = True)

        model_to_train.compile(optimizer = keras.optimizers.SGD(learning_rate = learning_rate), 
                                                     loss = 'categorical_crossentropy', 
                                                     metrics = ['categorical_accuracy'])

        model_to_train.fit(X_train, y_train, 
                          epochs = epochs, 
                          batch_size = batch_size,
                          #validation_split = 0.2,
                          validation_data = (X_test, y_test),
                          verbose = Verbosity, 
                          callbacks = [model_checkpoint])

        evaluation = model_to_train.evaluate(X_test, y_test)
        print("Test Accuracy = ", evaluation[1])

        y_pred = model_to_train.predict(X_test, verbose = 1)
        confusion_matrix = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1))
        
        print("Confusion Matrix for Training Set Size " + str(training_set_size[i]), confusion_matrix)

        accuracies.append(evaluation[1] * 100)

        # model to be saved
        model_to_train._layers.pop()
        model_to_train._layers.pop()

        sub_model =  Model(model_to_train.inputs, model_to_train.layers[-1].output)
        sub_model.compile(optimizer=keras.optimizers.SGD(lr = 0.001, decay = 1e-5, momentum = 0.9, nesterov = True),
                          loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy'])
        sub_model.set_weights(model_to_train.get_weights())
        sub_model.summary()

        sub_model.save('/content/drive/My Drive/Hyperspectral_Image_Classification/code//Trained_models//sub_models//' + source + '_with_' + str(int(training_set_size[i])) + '_samples_from_each_class_in_training_set.h5')

                        
    return accuracies

def transfer_learning(source_dataset,
                      target_dataset,
                      data,
                      ground_truth,
                      source_training_size,
                      training_samples_from_each_class,
                      test_samples_from_each_class,
                      classes,
                      overlap_ratio,
                      channels,
                      cube_size,
                      learning_rate,
                      epochs,
                      batch_size
                      ):

    pretrained_model = load_model('/content/drive/My Drive/Hyperspectral_Image_Classification/code/Trained_models/sub_models/' + source_dataset + '_with_' + str(int(source_training_size)) +
                              '_samples_from_each_class_in_training_set.h5')

    pretrained_model.summary()


    X_train, X_test, y_train, y_test, counts_test_set, class_samples = sample_extraction_from_botswana(classes = classes, 
                                                                                                        cube_size = cube_size, 
                                                                                                        data = data, 
                                                                                                        ground_truth = ground_truth, 
                                                                                                        cubes = [], 
                                                                                                        output_class = [], 
                                                                                                        training_samples_from_each_class = training_samples_from_each_class,
                                                                                                        test_samples_from_each_class = test_samples_from_each_class,
                                                                                                        overlap_ratio = overlap_ratio, 
                                                                                                        channels = channels)
    X_train_transfer = pretrained_model.predict(X_train)
    X_test_transfer = pretrained_model.predict(X_test)
    
    print('X_train_transfer => ' + str(X_train_transfer.shape) + '\n' +
          'X_test_transfer  => ' + str(X_test_transfer.shape) + '\n' +
          'y_train => ' + str(y_train.shape) + '\n' +
          'y_test  => ' + str(y_test.shape) + '\n')

    fine_tune_on_target = fine_tune_target(input_shape = X_train_transfer[0].shape, classes = len(y_train[0]))
    fine_tune_on_target.summary()

    model_checkpoint = ModelCheckpoint('/content/drive/My Drive/Hyperspectral_Image_Classification/code/Trained_models/transferred_models/fine_tune_on_'
                                       + target_dataset + '_with_' + str(int(training_samples_from_each_class)) + '_samples_from_each_class.h5',
                                       monitor = 'val_categorical_accuracy', 
                                       verbose = 1, 
                                       save_best_only = True)

    fine_tune_on_target.compile(optimizer = keras.optimizers.SGD(lr = learning_rate, 
                                                          decay = 1e-5, 
                                                          momentum = 0.9, 
                                                          nesterov = True),
                                                          loss = 'categorical_crossentropy', 
                                                          metrics = ['categorical_accuracy'])

    fine_tune_on_target.fit(X_train_transfer, y_train, 
                             epochs = epochs, 
                             batch_size = batch_size,
                             validation_data = (X_test_transfer, y_test), 
                             verbose = 1, 
                             callbacks = [model_checkpoint])

    evaluation = fine_tune_on_target.evaluate(X_test_transfer, y_test)
    
    print("Test accuracy on target dataset = " + str(evaluation[1]))
    
    y_pred = fine_tune_on_target.predict(X_test_transfer, verbose = 1)
    
    confusion_matrix = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1))

    confusion_matrix = pd.DataFrame(confusion_matrix)

    confusion_matrix.columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    confusion_matrix.index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

    # append total samples per class in test set to the confusion matrix
    confusion_matrix = confusion_matrix.append(pd.DataFrame(counts_test_set.reshape(1,-1), columns = list(confusion_matrix)), ignore_index = True)

    confusion_matrix = confusion_matrix.rename(index = {confusion_matrix.index[-1]: 'Total Samples'})

    # extract correct predictions from confusion matrix
    correct_predictions = np.diag(confusion_matrix)

    # get accuracies for each class
    classification_accuracies = np.round((correct_predictions / counts_test_set) * 100, 2)
    
    classification_accuracies = np.append(classification_accuracies, '-')

    confusion_matrix['classfication_accuracies'] = classification_accuracies
    
    return confusion_matrix
        
    
