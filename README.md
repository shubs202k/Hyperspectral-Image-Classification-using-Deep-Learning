# Hyperspectral-Image-Classification-using-Deep-Learning
Hyperspectral Image Classification using Deep Neural Network Architectures with Transfer Learning

This is an attempt to implement SGCNN-X (Shuffled Group Convolutional Neural Network) models from the paper https://www.mdpi.com/2072-4292/12/11/1780 where X represents the number of convolution layers. 

To classify Hypersectral Images using transfer learning, following steps are executed
(Source : Indian Pines, Target : Botswana)

1. Samples of size **S X S X 64** (S - sample size,) from the image and labels are assigned to those samples using ground truth image. Samples are extracted using a 
   variable overlap_ratio which leads to generation of several datasets. An overlap ratio of 25% implies that a sample belonging to a class will be selected if and 
   only if the next sample from same class does not overlap more than 25% for this sample.
   
   Sample size (cube size) chosen is 20 (19 in paper)
   
   Center pixel from the ground truth image is used to assign label to a sample.
   
   An overlap ratio of 1 generates maximum number of samples.

2. Once the samples are extracted, two methods below have been used to split the extracted samples into training & test sets.
   
   **V1** : Specific number of samples from each class are put into the training set for pretraining (source) as well as fine tuning (target). 
            This methods follows the train test split tables mentioned in the paper.
        
   **Other** : Some percentage of samples are put into the training set and the rest are added to the test set.
               Models in folder SGCNN_7, SGCNN_8 & SGCNN_12 use this train test split method.
   
## Folder Details

Datasets : mat files of image and ground truth for Indian Pines, Botswana & Pavia datasets.

V1 : Implementation of SGCNN with traning and test set split as mentioned in the paper.

SGCNN_7 : Implementation of SGCNN-7 architecture from the paper. Model is trained on Indian Pines and fine tuned on Botswana.

SGCNN_8 : Implementation of SGCNN-8 architecture from the paper. Model is trained on Indian Pines and fine tuned on Botswana.

SGCNN_12 : Implementation of SGCNN-12 architecture from the paper. Model is trained on Indian Pines and fine tuned on Botswana.
