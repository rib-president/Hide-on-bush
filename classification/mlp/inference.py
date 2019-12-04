from keras.models import model_from_json
from keras.utils import np_utils
from collections import Counter
import tensorflow as tf
import keras
import numpy as np
import collections
import random
import csv

modelname = 'model'


def load_mlp_model(modelname):
    json_file = open(modelname + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(modelname + ".h5")
    loaded_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    return loaded_model
    
def predict(fp_array, mlp_model):    
    max_score = 0
    cnt = 0
    # mlp input: numpy array - [[xc, yc, w, h, p][xc, yc, w, h, p]...]
    for fp in fp_array:
        current = np.array(fp)
        prediction = mlp_model.predict_proba(current) 
        max_pred = np.argmax(prediction)
        max_score = prediction[0][max_pred]

        print(fp, max_pred, max_score)

        if fp[0].index(max(fp[0])) == max_pred:
            cnt += 1
    print 'accuracy', cnt/1000.0*100
    
if __name__ == '__main__':
    mlp_model = load_mlp_model(modelname)
    test_path = './test.csv'
    fp_array = []
    with open(test_path, 'r') as df:
        read = csv.reader(df, delimiter=',')
        for i, row in enumerate(read):
            fp_array.append([row])
    
    predict(fp_array, mlp_model)
