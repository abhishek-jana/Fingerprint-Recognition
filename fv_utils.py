# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:32:09 2019

@author: abhishek
"""

# Example 
#filepath = '../FVC2006Dbs/Dbs/DB1_A/1_1.bmp'

""" unitity functions """

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

def read_image(path):
    """ function to read single image at the given path
        note: the loaded image is in B G R format
    """
    return cv.imread(path)


def BGR2RGB(image):
    """ function to transform image from BGR into RBG format """
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def BGR2Gray(image):
    """ function to transofrm image from BGR into Gray format """
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def show_image(image, img_format='RGB', figsize=(8, 6)):
    """ function to show image """
    if img_format == 'RGB' or img_format == 'Gray':
        pass
    elif img_format == 'BGR':
        image = BGR2RGB(image)
    else:
        raise ValueError('format should be "RGB", "BGR" or "Gray"')

    fig, ax = plt.subplots(figsize=figsize)
    if format == 'Gray':
        ax.imshow(image, format='gray')
    else:
        ax.imshow(image)
    return fig


def detect_finger(image, face):
    """ function to denote location of finger on image """
    img = image.copy()
    for (x, y, w, h) in face:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img


def crop_finger(image, face, scale_factor=1.0, target_size=(128, 128)):
    """ crop finger at the given positons and resize to target size """
    rows, columns, channels = image.shape
    x, y, w, h = face[0]
    mid_x = x + w // 2
    mid_y = y + h // 2

    # calculate the new vertices
    x_new = mid_x - int(w // 2 * scale_factor)
    y_new = mid_y - int(h // 2 * scale_factor)
    w_new = int(w * scale_factor)
    h_new = int(h * scale_factor)

    # validate the new vertices
    left_x = max(0, x_new)
    left_y = max(0, y_new)
    right_x = min(columns, x_new + w_new)
    right_y = min(rows, y_new + h_new)

    # crop and resize the facial area
    cropped = image[left_y:right_y, left_x:right_x, :]
    resized = cv.resize(cropped, dsize=target_size, interpolation=cv.INTER_LINEAR)

    return resized

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    images = np.array(images)
    return images

def img_to_encoding(image_path, model):
    img1 = cv.imread(image_path, 1)
    img = img1[...,::-1]
    img = np.around(img/255.0, decimals=12)
    #img = np.around(img/255.0)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

def get_data(path):
    data = {}
    for files in os.listdir(path):
        keys,_ = files.split('_')
        if keys in data:
            data[keys].append(files)
        else:
            data[keys] = [files]
    return data

def train_test_split(data, ratio = 0.2):
    train = {}
    test = {}
    for key in data.keys():
        vals = data[key]
        split = int(len(vals)*ratio)
        train[key] = vals[split:]
        test[key] = vals[:split]
    return train,test

def get_data_label(path,ratio = 0.2):
    """
    Given path returns tran and test images and label associated with it
    """
    _data = get_data(path)
    _train,_test = train_test_split(_data, ratio = ratio)
    train_image = []
    train_labels = []
    test_image = []
    test_labels = []
    for keys, vals in _train.items():
        train_image += [[cv.imread(os.path.join(path,files))/255. \
                        for files in vals]]
        train_labels += [keys]
        
    for keys, vals in _test.items():
        test_image += [[cv.imread(os.path.join(path,files))/255. \
                       for files in vals]]
        test_labels += [keys]  
    return np.array(train_image), np.array(train_labels), \
np.array(test_image), np.array(np.array(test_labels)),_test

#def load_data():
    

