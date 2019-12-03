from __future__ import print_function
import os,sys
import keras
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop , Adagrad , Adam , Adamax
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


model_dir = './saved_model/'
model_name = 'multi_checkpoint-0010' + '.h5' 

dense_num = 512
optimizer_name = 'adamax'

class_num = 2458

model = Sequential()
model.add(Dense(dense_num, kernel_initializer='uniform', activation='relu', input_dim=2048))
model.add(Dropout(0.2))
model.add(Dense(class_num, kernel_initializer='uniform', activation='softmax'))
print (model)
model.load_weights(model_dir + model_name)
model.compile(optimizer=optimizer_name, loss="categorical_crossentropy", metrics=["accuracy"])

model_json = model.to_json()
with open('./saved_model/' + model_name + '.json', 'w') as json_file:
    json_file.write(model_json)
    
    
print("complete to json")
