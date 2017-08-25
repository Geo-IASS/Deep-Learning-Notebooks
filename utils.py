
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from glob import glob
import bcolz
import matplotlib.pyplot as plt
import itertools
from itertools import chain

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]

def plots(images,title=None):
    for i in range(0,len(images),5):
        fig = plt.figure(figsize=(12,6))
        l = len(images)
        for j in range(5):
            ax = fig.add_subplot(1,5,j+1)
            ax.set_axis_off()
            if (i+j) >= l:break
            if not title is None:ax.set_title(title[i+j])
            ax.imshow(images[i+j],cmap='gray')
def plot(img,interp=False):
    f = plt.figure(figsize=(3,6),frameon=True)
    plt.imshow(img,interpolation =None if interp else "none",cmap='gray')
    plt.show()
    
def get_batches(path,gen=image.ImageDataGenerator(),shuffle=True,batch_size=64
                ,target_size=(224,224)
                ,class_mode='categorical'):
        return gen.flow_from_directory(path,target_size,batch_size=batch_size
                                       ,shuffle=shuffle
                                       ,class_mode=class_mode)

def get_data(path):
    batches = get_batches(path,batch_size=1,shuffle=False,class_mode=None)
    return np.concatenate([batches.next() for i in range(batches.n)])

def finetune(vgg,num,train_layers=None):
    layers = vgg.layers[:-1]
    if train_layers is None:
        for l in layers: l.trainable = False
    else:
        for l in layers[:-train_layers]: l.trainable = False
    
    model = Sequential(layers)
    model.add(Dense(2,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def fit(model,train_batches,valid_batches,epochs):
    return model.fit_generator(train_batches,samples_per_epoch=train_batches.n
                        ,validation_data=valid_batches
                        ,nb_val_samples=valid_batches.n
                        ,nb_epoch=epochs
                        ,nb_worker=4, pickle_safe=True)
def save_array(arr,path):
    c = bcolz.carray(arr,rootdir=path,mode='w')
    c.flush()
    
def load_array(path):
    return bcolz.open(path,mode='r')[:]

def one_hot(arr,classes):
    return to_categorical(arr,classes)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    





