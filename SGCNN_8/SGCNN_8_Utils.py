## Required Imports
import scipy.io as sio
import math
import sklearn.metrics
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

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

# SGCNN Model Architecture

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

def channel_shuffle(X):

    cardinality = 8
    
    b, h, w, c = X.shape
    
    X = tf.reshape(X, [-1, h, w, cardinality, c // cardinality])
    X = tf.transpose(X, perm = [0, 1, 2, 4, 3])
    X = tf.reverse(X,[-1])
    X = tf.reshape(X, [-1, h, w, c])
    
    return X

def SG_Unit(X):
    
    # Save the input value
    X_shortcut = X
    l2_ = 0.01
    X = Conv2D(64, (1, 1), kernel_regularizer = regularizers.l2(l2_), padding = "same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Group convolution
    X = group_convolution(X, 64)

    # channel shuffle
    X = channel_shuffle(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(128, (1, 1), kernel_regularizer = regularizers.l2(l2_), padding = "same")(X)
    X = BatchNormalization()(X)
    
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

## Sample extraction & train test split
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
        
    return cubes, output_class, class_samples

## Prepare Training & Test Data

def train_test_split_using_percentage(percentage, class_samples, cubes, output_class):

    '''
    Put a percentage of samples from each class in the training set and put the rest of the samples in the test set.

    percentage : Percentage of samples to put in training set from each class.
    class_samples : List with number os samples extracted for each class.
    cubes : List of extracted samples.
    output_class : List with output class of each extracted sample.
    
    '''
    
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    class_division = [0]

    c = 0

    for i in range(len(class_samples)):

        class_division.append(int(class_samples[i] * (percentage / 100)) + c)
        class_division.append(class_samples[i] + c)
        c = class_samples[i] + c

    for i in range(1, len(class_division)):
        
        if i % 2 != 0:
            
            for j in range(class_division[i - 1], class_division[i]):
                X_train.append(cubes[j])
                y_train.append(output_class[j])
        else:
            for k in range(class_division[i - 1], class_division[i]):
                X_test.append(cubes[k])
                y_test.append(output_class[k])
                
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # shuffle training set
    sampled_train = np.arange(X_train.shape[0])
    np.random.shuffle(sampled_train)
    X_train = X_train[sampled_train]
    y_train = y_train[sampled_train]

    # shuffle test set
    sampled_test = np.arange(X_test.shape[0])
    np.random.shuffle(sampled_test)
    X_test = X_test[sampled_test]
    y_test = y_test[sampled_test]

    # Get counts(samples) of each class in test set
    values_test_set, counts_test_set = np.unique(y_test, return_counts = True)
    values_training_set, counts_training_set = np.unique(y_train, return_counts = True)

    samples_in_training_set = np.sum(counts_training_set)
    samples_in_test_set = np.sum(counts_test_set)

    print("Samples per class: " + str(class_samples) + '\n'
          "Total number of samples is " + str(np.sum(class_samples)) + '.\n')
    
    print("unique classes in training set: " + str(values_training_set) + '\n'
          "Total number of samples in training set is " + str(np.sum(counts_training_set)) + '.\n'
          "Samples per class in training set: " + str(counts_training_set) + '\n')

    print("unique classes in test set: " + str(values_test_set) + '\n'
          "Total number of samples in test set is " + str(np.sum(counts_test_set)) + '.\n'
          "Samples per class in test set: " + str(counts_test_set) + '\n')

    ## one hot encode labels
    onehot_encoder = OneHotEncoder(sparse = False)

    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)

    y_train = onehot_encoder.fit_transform(y_train)
    y_test = onehot_encoder.fit_transform(y_test)

    return X_train, X_test, y_train, y_test, samples_in_training_set, samples_in_test_set, counts_test_set, class_samples

def sample_extraction(classes,
                      percentage,
                      cube_size,
                      data,
                      ground_truth,
                      cubes,
                      output_class,
                      overlap_ratio,             
                      channels):

    '''
    Extract samples using the two functions collect_samples_from_all_classes &
    train_test_split_using_percentage.

    Output:
    X_train, X_test, y_train, y_test
    samples_in_training_set : Total samples in the training set.
    samples_in_test_set : Total samples in test set.
    counts_test_set : test samples for each class.
    class_samples : samples for each class.
    '''
    
    cubes, output_class, class_samples = collect_samples_from_all_classes(classes,
                                                                          cube_size, 
                                                                          data,  
                                                                          ground_truth, 
                                                                          cubes, 
                                                                          output_class , 
                                                                          overlap_ratio, 
                                                                          channels)
    
    X_train, X_test, y_train, y_test, samples_in_training_set, samples_in_test_set, counts_test_set, class_samples = train_test_split_using_percentage(percentage,
                                                                                                                                                       class_samples, 
                                                                                                                                                       cubes,
                                                                                                                                                       output_class)
                                                                                                            
    
    return X_train, X_test, y_train, y_test, samples_in_training_set, samples_in_test_set, counts_test_set, class_samples

def pretrain_source_models(percentages,
                           classes,
                           cube_size,
                           overlap_ratios,
                           data,
                           ground_truth,
                           batch_size,
                           channels,
                           epochs,
                           Verbosity,
                           accuracies,
                           learning_rate,
                           source_dataset):

    '''
    Train SGCNN8 model on source dataset. Save the model except the last two layers. The saved model will be used by transferring it
    to a target dataset for fine tuning and classification.

    percentages : list with percentages used to split samples into training and test sets.
    classes : output classes
    cube_size : size of extracted sample.
    overlap_ratios : overlap ratio used while extracting samples.
    data : source data.
    ground_truth : output labels.
    batch_size : batch size while training.
    channels : channels used.
    epochs : epochs used for training.
    accuracies : list with test accuracy for each model.
    learning_rate : learning rate for training.
    source_dataset : name of source dataset.
    
    '''
    overlap = []
    train_test_split = []
    training_samples = []
    test_samples = []
    
    # loop over different overlap ratios used to extract samples 
    for i in range(len(overlap_ratios)):
        
        # loop over different percentage used for training and test set split.

        for j in range(len(percentages)):

            
            print("\n=============================================================================================================\n"
                  "Model training starts for data with overlap ratio " + str(float(overlap_ratios[i])) +
                  " and " + str(int(percentages[j])) + " percent samples from each class in training set \n"
                  "==============================================================================================================\n")



            X_train, X_test, y_train, y_test, samples_in_training_set, samples_in_test_set, counts_test_set, class_samples = sample_extraction(percentage = percentages[j],
                                                                                                                                              classes = classes,
                                                                                                                                              cube_size = cube_size,
                                                                                                                                              data = data,
                                                                                                                                              ground_truth = ground_truth,
                                                                                                                                              cubes = [],
                                                                                                                                              output_class = [],
                                                                                                                                              overlap_ratio = overlap_ratios[i],             
                                                                                                                                              channels = channels)
            
            print('X_train => ' + str(X_train.shape) + '\n' +
                  'X_test  => ' + str(X_test.shape) + '\n' +
                  'y_train => ' + str(y_train.shape) + '\n' +
                  'y_test  => ' + str(y_test.shape) + '\n')

            X_train = np.array(X_train).astype(np.float32)
            X_test = np.array(X_test).astype(np.float32)

            train_on_source = model(input_shape = X_train[0].shape, classes = len(classes))
            train_on_source.summary()

            # save best model
            early_stopping_callback = EarlyStopping(monitor = 'val_categorical_accuracy', patience = 2, mode = 'max')
            model_checkpoint = ModelCheckpoint('/content/drive/My Drive/Hyperspectral_Image_Classification/SGCNN_8//Trained_models//full_models//' + source_dataset +
                                               '_as_source_with_overlap_ratio_' + str(float(overlap_ratios[i])) + '_and_' + str(int(percentages[j])) + '_samples_from_each_class_in_training_set.h5',
                                                monitor = 'val_categorical_accuracy', 
                                                verbose = 1, 
                                                save_best_only = True,
                                                mode = 'max')

            train_on_source.compile(optimizer = keras.optimizers.SGD(learning_rate = learning_rate, decay = 1e-5, momentum = 0.9, nesterov = True), 
                                    loss = 'categorical_crossentropy', 
                                    metrics = ['categorical_accuracy'])

            train_on_source.fit(X_train, y_train, 
                              epochs = epochs, 
                              batch_size = batch_size,
                              #validation_split = 0.2,
                              validation_data = (X_test, y_test),
                              verbose = Verbosity, 
                              callbacks = [early_stopping_callback, model_checkpoint])

            # load checkpointed(best) model for evaluation on test set
            model_for_evaluation = load_model('/content/drive/My Drive/Hyperspectral_Image_Classification/SGCNN_8//Trained_models//full_models//' + source_dataset +
                                               '_as_source_with_overlap_ratio_' + str(float(overlap_ratios[i])) + '_and_' + str(int(percentages[j])) + '_samples_from_each_class_in_training_set.h5')

            # evaluate on test set
            evaluation = model_for_evaluation.evaluate(X_test, y_test)

            print("Test Accuracy = ", np.round(evaluation[1] * 100))

            y_pred = model_for_evaluation.predict(X_test, verbose = 1)
            confusion_matrix = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1))
            
            #print("Confusion Matrix", confusion_matrix

            # for results
            accuracies.append(np.round(evaluation[1] * 100,2))
            overlap.append(overlap_ratios[i])
            train_test_split.append(percentages[j])
            training_samples.append(samples_in_training_set)
            test_samples.append(samples_in_test_set)

            # drop last two layers
            model_for_evaluation._layers.pop()
            model_for_evaluation._layers.pop()

            # model for transfer learning
            sub_model =  Model(model_for_evaluation.inputs, model_for_evaluation.layers[-1].output)
            
            sub_model.compile(optimizer=keras.optimizers.SGD(learning_rate = learning_rate, decay = 1e-5, momentum = 0.9, nesterov = True),
                              loss = 'categorical_crossentropy',
                              metrics = ['categorical_accuracy'])
            
            sub_model.set_weights(model_for_evaluation.get_weights())

            sub_model.summary()

            sub_model.save('/content/drive/My Drive/Hyperspectral_Image_Classification/SGCNN_8//Trained_models//sub_models//' + source_dataset +
                           '_as_source_with_overlap_ratio_' + str(float(overlap_ratios[i])) + '_and_' + str(int(percentages[j])) + '_samples_from_each_class_in_training_set.h5')

            print("\n=============================================================================================================\n"
                  "\n=============================================================================================================\n"
                  "\n=============================================================================================================\n"
                  "\n=============================================================================================================\n")
        

    pretrain_results = pd.DataFrame(list(zip(overlap, training_samples, test_samples, train_test_split, accuracies)))
    pretrain_results.columns = ['Overlap_ratio', 'Training Samples', 'Test Samples', 'Training_Test_Split','Test_Accuracies']

                        
    return pretrain_results

def transfer_learning(percentages,
                      source_dataset,
                      target_dataset,
                      data,
                      ground_truth,
                      classes,
                      overlap_ratios,
                      channels,
                      cube_size,
                      learning_rate,
                      epochs,
                      batch_size
                      ):


    '''
    Load model trained on source dataset.
    Fine tune on samples extracted from target dataset.
    Evaluate model on test set and get confusion matrix and classification accuracies for each class.
    
    '''
    overlap = []
    train_test_split = []
    training_samples = []
    test_samples = []
    test_accuracies = []
    confusion_matrixes = []
    
    for i in range(len(overlap_ratios)):
        
        for j in range(len(percentages)):

            print("\n=============================================================================================================\n"
                  "Model training starts for data with overlap ratio " + str(float(overlap_ratios[i])) +
                  " and " + str(int(percentages[j])) + " percent samples from each class in training set \n"
                  "==============================================================================================================\n")
            
            # load pretrained model
    
            pretrained_model = load_model('/content/drive/My Drive/Hyperspectral_Image_Classification/SGCNN_8//Trained_models//sub_models//' + source_dataset +
                               '_as_source_with_overlap_ratio_' + str(float(overlap_ratios[i])) + '_and_' + str(int(percentages[j])) + '_samples_from_each_class_in_training_set.h5')

            pretrained_model.summary()


            X_train, X_test, y_train, y_test, samples_in_training_set, samples_in_test_set, counts_test_set, class_samples = sample_extraction(percentage = percentages[j],
                                                                                                                                               classes = classes, 
                                                                                                                                               cube_size = cube_size, 
                                                                                                                                               data = data, 
                                                                                                                                               ground_truth = ground_truth, 
                                                                                                                                               cubes = [], 
                                                                                                                                               output_class = [], 
                                                                                                                                               overlap_ratio = overlap_ratios[i], 
                                                                                                                                               channels = channels)
            X_train_transfer = pretrained_model.predict(X_train)
            X_test_transfer = pretrained_model.predict(X_test)
            
            print('X_train_transfer => ' + str(X_train_transfer.shape) + '\n' +
                  'X_test_transfer  => ' + str(X_test_transfer.shape) + '\n' +
                  'y_train => ' + str(y_train.shape) + '\n' +
                  'y_test  => ' + str(y_test.shape) + '\n')

            fine_tune_on_target = fine_tune_target(input_shape = X_train_transfer[0].shape, classes = len(y_train[0]))
            fine_tune_on_target.summary()

            model_checkpoint = ModelCheckpoint('/content/drive/My Drive/Hyperspectral_Image_Classification/SGCNN_8//Trained_models//transferred_models/fine_tune_on_'
                                               + target_dataset + '_with_overlap_ratio_' + str(float(overlap_ratios[i])) + '_and_' + str(int(percentages[j])) + '_samples_from_each_class_in_training_set.h5',
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

            # load checkpointed model for evaluation

            model_for_evaluation = load_model('/content/drive/My Drive/Hyperspectral_Image_Classification/SGCNN_8//Trained_models//transferred_models/fine_tune_on_'
                                               + target_dataset + '_with_overlap_ratio_' + str(float(overlap_ratios[i])) + '_and_' + str(int(percentages[j])) + '_samples_from_each_class_in_training_set.h5')

            
            evaluation = model_for_evaluation.evaluate(X_test_transfer, y_test)
            
            print("Test accuracy on target dataset = " + str(evaluation[1]))
            
            y_pred = model_for_evaluation.predict(X_test_transfer, verbose = 1)

            # confusion matrix
            confusion_matrix = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis = 1), np.argmax(y_pred, axis = 1))

            confusion_matrix = pd.DataFrame(confusion_matrix)

            # columns & indexes for dataframe equal to number of classes in test set.
            confusion_matrix.columns = [num for num in range(1,len(counts_test_set) + 1)]
            confusion_matrix.index = [num for num in range(1,len(counts_test_set) + 1)]

            # append total samples per class in test set to the confusion matrix
            confusion_matrix = confusion_matrix.append(pd.DataFrame(counts_test_set.reshape(1,-1), columns = list(confusion_matrix)), ignore_index = True)

            confusion_matrix = confusion_matrix.rename(index = {confusion_matrix.index[-1]: 'Total Samples'})

            # extract correct predictions from confusion matrix
            correct_predictions = np.diag(confusion_matrix)

            # get accuracies for each class
            classification_accuracies = np.round((correct_predictions / counts_test_set) * 100, 2)
            
            classification_accuracies = np.append(classification_accuracies, '-')

            confusion_matrix['classfication_accuracies'] = classification_accuracies

            confusion_matrixes.append(confusion_matrix)
            # for results
            test_accuracies.append(np.round(evaluation[1] * 100 ,2))
            overlap.append(overlap_ratios[i])
            train_test_split.append(percentages[j])
            training_samples.append(samples_in_training_set)
            test_samples.append(samples_in_test_set)

            print("\n=============================================================================================================\n"
                  "\n=============================================================================================================\n"
                  "\n=============================================================================================================\n"
                  "\n=============================================================================================================\n")

    transfer_results = pd.DataFrame(list(zip(overlap, training_samples, test_samples, train_test_split, test_accuracies)))
    transfer_results.columns = ['Overlap_ratio', 'Training Samples', 'Test Samples', 'Training_Test_Split','Test_Accuracies']

            
    return transfer_results, confusion_matrixes

