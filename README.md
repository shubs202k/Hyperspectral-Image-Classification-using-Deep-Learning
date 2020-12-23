# Hyperspectral-Image-Classification-using-Deep-Learning
Hyperspectral Image Classification using Deep Neural Network Architectures with Transfer Learning

This is an attempt to implement SGCNN(Shuffled Group Convolutional Neural Network) models from the paper https://www.mdpi.com/2072-4292/12/11/1780.

To classify Hypersectral Images using transfer learning, following steps are executed
(Source : Indian Pines, Target : Botswana)

1. Samples of size **S X S X 64** from the image and labels are assigned to those samples using ground truth image. Samples are extracted using a variable
   overlap_ratio which leads to generation of several datasets. An overlap ratio of 25% implies that a sample belonging to a class will be selected if and only if
   the next sample from same class does not overlap more than 25% for this sample.
   
   Center pixel from the ground truth image is used to assign label to a sample.
   
   An overlap ratio of 1 generates maximum number of samples.

2. Once the samples are extracted, two methods below have been used to split the extracted samples into training & test sets.
   
   **V1** : Specific number of samples from each class are put into the training set. This methods follows the train 
            test split tables mentioned in the paper.
        
   **V2** : Some percentage of samples are put into the training set and the rest are added to the test set.
            Models in folder SGCNN_7, SGCNN_8 & SGCNN_12 use this train test split method.
   
## Folder Details

Datasets : mat files of image and ground truth for Indian Pines, Botswana & Pavia datasets.

V1 : Implementation of SGCNN with traning and test set split as mentioned in the paper.

SGCNN_7 : Implementation of SGCNN-7 architecture from the paper. Model is trained on Indian Pines and fine tuned on Botswana.

SGCNN_8 : Implementation of SGCNN-8 architecture from the paper. Model is trained on Indian Pines and fine tuned on Botswana.

SGCNN_12 : Implementation of SGCNN-12 architecture from the paper. Model is trained on Indian Pines and fine tuned on Botswana.

SGCNN_8_Utils.py, SGCNN_7_Utils.py & SGCNN_12_Utils.py : Python code that builds respective model architectures, sample extraction and train test split functions.
                                                         Involves shuffled group convolution units that do group convolution based on a variable cardinality(paths                                                            along which channels are split into and convolution is done seperately for all paths) along with the                                                                channel shuffle operation. Includes models used pretraining on source and finetuning on target.

sample_extraction_V1_utils.py : Python code for extracting samples with overlap ratio. Train test split for different datasets as mentioned in paper.
