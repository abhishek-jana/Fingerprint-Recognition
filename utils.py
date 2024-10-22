import numpy as np
import os


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

def get_label(path):
    """
    Given path returns image and label
    """
    image = []
    labels = []
    for files in os.listdir(path):
        label,_ = files.split('_')
        labels.append(label)
        image.append(file)
    return image,label

