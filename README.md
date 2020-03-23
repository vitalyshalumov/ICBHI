#  Classification by CNN using MFCC features using ICBHI database.
Based on https://www.kaggle.com/gizemtanriver/disease-classification-by-cnn-using-mfcc


                                                    
# Overview

The model will be a Convolutional Neural Network (CNN) using Keras and a Tensorflow backend.

A sequential model is used, with a simple model architecture, consisting of four Conv2D convolution layers, with the final output layer being a dense layer.

The first layer will receive the input shape of (40, 862, 1) where 40 is the number of MFCC's, 862 is the number of frames taking padding into account and the 1 signifying that the audio is mono.

A small Dropout value of 20% is used on the convolutional layers.

Each convolutional layer has an associated pooling layer of MaxPooling2D type with the final convolutional layer having a GlobalAveragePooling2D type. 

The output layer will has 6 nodes (num_labels) which matches the number of possible classifications.
