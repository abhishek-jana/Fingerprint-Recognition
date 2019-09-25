#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:30:01 2019

@author: Abhishek Jana
"""

# Importing the libraries
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D,Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

def conv_block(x, nb_filter, nb_row, nb_col, padding = "same", strides = (1, 1), use_bias = False):
    '''Defining a Convolution block that will be used throughout the network.'''
    
    x = Conv2D(nb_filter, (nb_row, nb_col), strides = strides, padding = padding, use_bias = use_bias)(x)
    x = BatchNormalization(axis = -1, momentum = 0.9997, scale = False)(x)
    x = Activation("relu")(x)
    
    return x

def deconv_block(x, nb_filter, nb_row, nb_col, padding = "valid", strides = (1, 1), use_bias = False):
    '''Defining a Deconvolution block that will be used throughout the network.'''
    
    x = Conv2DTranspose(nb_filter, (nb_row, nb_col), strides = strides, padding = padding, use_bias = use_bias)(x)
    x = BatchNormalization(axis = -1, momentum = 0.9997, scale = False)(x)
    x = Activation("relu")(x)

def stem(input):
    '''The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.'''
    
    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    x = conv_block(input, 32, 3, 3, strides = (2, 2), padding = "valid") # 149 * 149 * 32
    x = conv_block(x, 32, 3, 3, padding = "valid") # 147 * 147 * 32
    x = conv_block(x, 64, 3, 3) # 147 * 147 * 64

    x1 = MaxPooling2D((3, 3), strides = (2, 2), padding = "valid")(x)
    x2 = conv_block(x, 96, 3, 3, strides = (2, 2), padding = "valid")

    x = concatenate([x1, x2], axis = -1) # 73 * 73 * 160

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, padding = "valid")

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 96, 3, 3, padding = "valid")

    x = concatenate([x1, x2], axis = -1) # 71 * 71 * 192

    x1 = conv_block(x, 192, 3, 3, strides = (2, 2), padding = "valid")
    
    x2 = MaxPooling2D((3, 3), strides = (2, 2), padding = "valid")(x)

    x = concatenate([x1, x2], axis = -1) # 35 * 35 * 384
    
    return x

def inception_A(input):
    '''Architecture of Inception_A block which is a 35 * 35 grid module.'''
    
    a1 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input)
    a1 = conv_block(a1, 96, 1, 1)
    
    a2 = conv_block(input, 96, 1, 1)
    
    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)
    
    a4 = conv_block(input, 64, 1, 1)
    a4 = conv_block(a4, 96, 3, 3)
    a4 = conv_block(a4, 96, 3, 3)
    
    merged = concatenate([a1, a2, a3, a4], axis = -1)
    
    return merged

def inception_B(input):
    '''Architecture of Inception_B block which is a 17 * 17 grid module.'''
    
    b1 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input)
    b1 = conv_block(b1, 128, 1, 1)
    
    b2 = conv_block(input, 384, 1, 1)
    
    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 256, 1, 7)
    
    b4 = conv_block(input, 192, 1, 1)
    b4 = conv_block(b4, 192, 1, 7)
    b4 = conv_block(b4, 224, 7, 1)
    b4 = conv_block(b4, 224, 1, 7)
    b4 = conv_block(b4, 256, 7, 1)
    
    merged = concatenate([b1, b2, b3, b4], axis = -1)
    
    return merged

def inception_C(input):
    '''Architecture of Inception_C block which is a 8 * 8 grid module.'''
    
    c1 = AveragePooling2D((3, 3), strides = (1, 1), padding = "same")(input)
    c1 = conv_block(c1, 256, 1, 1)
    
    c2 = conv_block(input, 256, 1, 1)

    c3 = conv_block(input, 384, 1, 1)
    c31 = conv_block(c2, 256, 1, 3)
    c32 = conv_block(c2, 256, 3, 1)
    c3 = concatenate([c31, c32], axis = -1)

    c4 = conv_block(input, 384, 1, 1)
    c4 = conv_block(c3, 448, 1, 3)
    c4 = conv_block(c3, 512, 3, 1)
    c41 = conv_block(c3, 256, 1, 3)
    c42 = conv_block(c3, 256, 3, 1)
    c4 = concatenate([c41, c42], axis = -1)
  
    merged = concatenate([c1, c2, c3, c4], axis = -1)
    
    return merged

def reduction_A(input, k = 192, l = 224, m = 256, n = 384):
    '''Architecture of a 35 * 35 to 17 * 17 Reduction_A block.'''

    ra1 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(input)
    
    ra2 = conv_block(input, n, 3, 3, strides = (2, 2), padding = "same")

    ra3 = conv_block(input, k, 1, 1)
    ra3 = conv_block(ra3, l, 3, 3)
    ra3 = conv_block(ra3, m, 3, 3, strides = (2, 2), padding = "same")

    merged = concatenate([ra1, ra2, ra3], axis = -1)
    
    return merged

def reduction_B(input):
    '''Architecture of a 17 * 17 to 8 * 8 Reduction_B block.'''
    
    rb1 = MaxPooling2D((3, 3), strides = (2, 2), padding = "same")(input)
    
    rb2 = conv_block(input, 192, 1, 1)
    rb2 = conv_block(rb2, 192, 3, 3, strides = (2, 2), padding = "same")
    
    rb3 = conv_block(input, 256, 1, 1)
    rb3 = conv_block(rb3, 256, 1, 7)
    rb3 = conv_block(rb3, 320, 7, 1)
    rb3 = conv_block(rb3, 320, 3, 3, strides = (2, 2), padding = "same")
    
    merged = concatenate([rb1, rb2, rb3], axis = -1)
    
    return merged

def inception_v4(nb_classes = 1001, load_weights = True):
    '''Creates the Inception_v4 network.'''
    
    init = Input((299, 299, 3)) # Channels last, as using Tensorflow backend with Tensorflow image dimension ordering
    
    # Input shape is 299 * 299 * 3
    x = stem(init) # Output: 35 * 35 * 384
    
    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)
        # Output: 35 * 35 * 384
        
    # Reduction A
    x = reduction_A(x, k = 192, l = 224, m = 256, n = 384) # Output: 17 * 17 * 1024

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)
        # Output: 17 * 17 * 1024
        
    # Reduction B
    x = reduction_B(x) # Output: 8 * 8 * 1536

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x) 
        # Output: 8 * 8 * 1536
        
    # Average Pooling
    x = AveragePooling2D((8, 8))(x) # Output: 1536

    # Dropout
    x = Dropout(0.2)(x) # Keep dropout 0.2 as mentioned in the paper
    x = Flatten()(x) # Output: 1536

    # Output layer
    output = Dense(units = nb_classes, activation = "softmax")(x) # Output: 1000

    model = Model(init, output, name = "Inception-v4")   
        
    return model

def inception_full(nb_classes = 256, load_weights = True):
    init = Input((299, 299, 3)) # Channels last, as using Tensorflow backend with Tensorflow image dimension ordering
    
    # Input shape is 299 * 299 * 3
    x1 = stem(init) # Output: 35 * 35 * 384
    x2 = x1
    
    # Texture Features
    
    # 4 x Inception A
    for i in range(4):
        x1 = inception_A(x1)

    # 7 x Inception B
    for i in range(7):
        x1 = inception_B(x1)


    # 3 x Inception C
    for i in range(3):
        x1 = inception_C(x1) 
        
    # Dropout
    x1 = Dropout(0.2)(x1) # Keep dropout 0.2 as mentioned in the paper
    x1 = Flatten()(x1) 
    
    # Output layer 1
    output1 = Dense(units = nb_classes, activation = "relu")(x1) # Output: 256 activation function??
    
    # Minutiae Features
    # sublayer 1 of layer 2 
    # 6 x Inception A
    for i in range(6):
        x2 = inception_A(x2) # 35 * 35 * 384
    
    x21 = conv_block(x2, 768, 1, 1) # 35 * 35 * 768
    x21 = conv_block(x21, 768, 1, 1, strides = (2, 2), padding = "valid") # 18 * 18 * 768
    x21 = conv_block(x21, 896, 1, 1) # 18 * 18 * 896
    x21 = conv_block(x21, 1024, 3, 3, strides = (2, 2)) # 9 * 9 * 1024 
    x21 = MaxPooling2D((1, 1))(x21) # 9 * 9 * 1024
    x21 = conv_block(x21, 1024, 9, 9, padding = "valid") # 1 * 1 * 1024
    x21 = Flatten()(x21) 
    
    # output layer 2
    output2 = Dense(units = nb_classes, activation = "relu")(x21) # Output: 256
    
    # Let's do sublayer 2 of output 2 later!
    
    
    model = Model(init, [output1,output2] , name = "Inception_full")
        
    return model

'''
if __name__ == "__main__":
    inception_txture = inception_txture()
    inception_txture.summary()
'''