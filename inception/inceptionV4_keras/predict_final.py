from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
from keras.models import model_from_json
import numpy as np
import os
import matplotlib.pyplot as plt



# load json and create model
json_file = open('/home/rib/iv4_test/saved_models_real_1121/model_test.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/home/rib/iv4_test/saved_models_real_1121/model_test.h5")
print("Loaded model from disk")
 
'''
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''


# data architecture   #
#data/inference/daisy #
#          /dandelion #
#          /......    #           

#get label
trainPath = 'data/train_dir'
train_array = sorted(os.listdir(trainPath))

#inference data
dataPath = 'data/inference'
dir_array = sorted(os.listdir(dataPath))


for directory in dir_array:
    dirPath = os.path.join(dataPath, directory)
    img_array = sorted(os.listdir(dirPath))

    for img_name in img_array:
        img_path = os.path.join(dirPath, img_name)
        img = image.load_img(img_path, target_size = (299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x/255, axis = 0)
        features = loaded_model.predict(x, verbose = 0)

        prob = features.tolist() 
        print prob

        y_pred = np.argmax(features, axis = 1)
        print 'prediction: ', y_pred, train_array[y_pred[0]], str(round(prob[0][y_pred[0]]*100 ,2)) +'%'

        plt.imshow(image.load_img(img_path))
        plt.text(0.02, 0.9, ('prediction: ' + train_array[y_pred[0]] + '   ' + str(round(prob[0][y_pred[0]]*100 ,2)) +'%'), fontsize=20, transform=plt.gcf().transFigure)
        plt.show()



