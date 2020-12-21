# Hyperspectral-Image-Classification-using-Deep-Learning
Hyperspectral Image Classification using Deep Neural Network Architectures with Transfer Learning

This is an attempt to implement SGCNN(Shuffled Group Convolutional Neural Network) models from the paper https://www.mdpi.com/2072-4292/12/11/1780.

To classify Hypersectral Images using transfer learning, following steps are executed
(Source : Indian Pines, Target : Botswana)

1. Extract samples from the image and assign labels using ground truth image. Samples are extracted using a variable overlap_ratio which leads
   to generation of several datasets. An overlap ratio of 25% implies that a sample belonging to a class will be selected if and only if the next 
   sample from same class does not overlap more than 25% for this sample.
   
   Center pixel from the ground truth image is used to assign label to a sample.
   
   An overlap ratio of 1 generates maximum number of samples.

2. Once the samples are extracted, two methods below have been used to split the extracted samples into training & test sets.
   
   **V1** : Specific number of samples from each class are put into the training set. This methods follows the train 
            test split tables mentioned in the paper.
        
   **V2** : Some percentage of samples are put into the training set and the rest are added to the test set.
   
## Folder Details

Datasets : mat files of image and ground truth for Indian Pines, Botswana & Pavia datasets.

V1 : Implementation of SGCNN with traning and test set split as mentioned in the paper.

V2 : Some percentage of samples into training set and rest in test set for datasets created with different overlap ratios.

sample_extraction_V1_utils.py : Python code for extracting samples with overlap ratio. Train test split for different datasets as mentioned in paper.

sample_extraction_V2_utils.py : Python code for extracting samples with overlap ratio. Train test split based on percentage of samples into training set.

SGCCN_8_model_design_utils.py : Python code that implements the model architecture. Involves shuffled group convolution units that do group convolution
                                based on a variable cardinality(paths along which channels are split into and convolution is done seperately for all 
                                paths) along with the channel shuffle operation. Includes models used pretraining on source and finetuning on target.
