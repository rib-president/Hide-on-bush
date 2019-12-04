from __future__ import print_function
import os,sys
import keras
from keras.models import Sequential
from keras.models import Model
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop , Adagrad , Adam , Adamax
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from keras.utils.training_utils import multi_gpu_model


start_time = time.time()
optimizer_name = 'adamax'
dense_num = 512
epoch_size = 10




class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)




def one_hot(y_):
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS



train_file = pd.read_csv('train.csv', delimiter=',', header=None)
test_file = pd.read_csv('val.csv', delimiter=',', header=None)
train_file = train_file.values
test_file = test_file.values

# number of rows in csv
train_rows = len(train_file)
print("train rows: %d"%(train_rows))
test_rows = len(test_file)
print("test_rows: %d"%(test_rows))

# [cell][row]
# 1:13 - 
X_train = train_file[:, 1:2049][:train_rows, :]
X_train = X_train.astype('float32')
y_train = train_file[:, :1][:train_rows, :]
y_train = one_hot(y_train)
print(X_train[0])
#print(y_train)
X_test = test_file[:, 1:2049][:test_rows, :]
X_test = X_test.astype('float32')
y_test = test_file[:, :1][:test_rows, :]
y_test = one_hot(y_test)

save_path = './saved_model'

if not os.path.exists(save_path):
    os.makedirs(save_path)


callbacks_list = [ModelCheckpoint(save_path + '/multi_checkpoint-{epoch:04d}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto'), EarlyStopping(monitor='val_loss', patience=50, verbose=0)]


# mlp model
model = Sequential()
model.add(Dense(dense_num, kernel_initializer='uniform', activation='relu', input_dim=2048))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], kernel_initializer='uniform', activation='softmax'))

parallel_model = ModelMGPU(model, gpus=4)
parallel_model.compile(optimizer=optimizer_name, loss="categorical_crossentropy", metrics=["accuracy"])

# train
hist=parallel_model.fit(X_train, y_train, nb_epoch=epoch_size, batch_size=512, validation_data=(X_test, y_test), callbacks=callbacks_list)


# eval
score = parallel_model.evaluate(X_test, y_test, batch_size=32)
print("Score of model: %f" %score[1])
print("\n%s: %.2f%%" % (parallel_model.metrics_names[1], score[1]*100))

# save model
filename = "adamax_multi_check"

model_json = model.to_json()


with open(save_path + "/" + filename + ".json", "w") as json_file:
    json_file.write(model_json)
    model.save_weights(save_path + "/" +filename + '.h5')
    
print("time: %s"%(time.time()-start_time))



