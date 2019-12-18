# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 01:22:14 2019

@author: abhis
"""
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers import GlobalAveragePooling2D,Dropout
from keras.layers.core import Lambda, Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


def fingerRecoModel(input_shape,embeddingsize):
    X_input = Input(input_shape)
    base=VGG16(weights='imagenet', input_tensor = X_input,input_shape = input_shape,include_top=False) 
    #imports the VGG16 model and discards the last 1000 neuron layer.
    X=base.output
    X=GlobalAveragePooling2D()(X)
    #x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    X=Dense(512,activation='relu')(X) #dense layer 2
    X=Dense(256,activation='relu')(X) #dense layer 3
    X=Dense(embeddingsize, name = 'dense_layer')(X)
    #x = Dropout(0.2)(x)
    # L2 normalization
    X = Lambda(lambda  x: K.l2_normalize(x,axis=1))(X)

    # Create model instance
    model = Model(inputs = X_input, outputs = X, name='FingerRecoModel')
        
    return model