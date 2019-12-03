"""
Author: Abner Ayala-Acevedo
This script based on examples provided in the keras documentation and a blog.
"Building powerful image classification models using very little data"
from blog.keras.io.
Dataset: Subset of Kaggle Dataset
https://www.kaggle.com/c/dogs-vs-cats/data
- cat pictures index 0-999 in data/train/cats
- cat pictures index 1000-1400 in data/validation/cats
- dogs pictures index 0-999 in data/train/dogs
- dog pictures index 1000-1400 in data/validation/dogs
Example: Dogs vs Cats (Directory Structure)
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
Example has 1000 training examples for each class, and 400 validation examples for each class.
The data folder already contains the dogs vs cat data you simply need to run script. For the dogs_cats classification
you can find a model already trained in the model folder. Feel free to create your own data.
"""

import sys
import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k
import keras
from inception_resnet_v2 import InceptionResNetV2
from keras.utils.training_utils import multi_gpu_model

'''
# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)
'''
# hyper parameters for model
nb_classes = 4014  # number of classes
based_model_last_block_layer_number = 126  # value is based on based model selected.
img_width, img_height = 299, 299  # change based on the shape/structure of your images
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 10  # number of iteration the algorithm gets trained.
learn_rate = 1e-4  # sgd learning rate
momentum = .9  # sgd momentum to avoid local minimum
transformation_ratio = .05  # how aggressive will be the data augmentation/transformation
nb_train_samples = nb_classes * 400  # Total number of train samples. NOT including augmented images
nb_validation_samples = nb_classes * 100  # Total number of train samples. NOT including augmented images.

img_width = 299
img_height = 299

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

def train(train_data_dir, validation_data_dir, model_path):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    
    base_model = InceptionResNetV2(include_top=False, pooling='avg')
    outputs = Dense(nb_classes, activation='softmax')(base_model.output)

    #x = Dropout(0.2)(x) #####################
    top_model = Model(base_model.inputs, outputs)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all layers of the based model that is already pre-trained.
    for layer in base_model.layers:
        layer.trainable = False

    # Read Data and Augment it: Make sure to select augmentations that are appropriate to your images.
    # To save augmentations un-comment save lines and add to your flow parameters.
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=[img_width, img_height],
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=[img_width, img_height],
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')

    #set multigpu
    parallel_model = ModelMGPU(top_model, gpus=4)

    parallel_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])

    # save weights of best training epoch: monitor either val_loss or val_acc

    top_weights_path = os.path.join(os.path.abspath(model_path), 'top_model_weights_test_finetune.h5')
    callbacks_list = [
        ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=0, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=5, verbose=0),
        keras.callbacks.TensorBoard(log_dir='tensorboard/inception-resnet-v2-train-top-layer', histogram_freq=0, write_graph=False, write_images=False)
    ]

    # Train Simple CNN
    parallel_model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=nb_epoch / 5,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        callbacks=callbacks_list)

    # verbose
    print("\nStarting to Fine Tune Model\n")

    # add the best weights from the train top model
    # at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
    # we re-load model weights to ensure the best epoch is selected and not the last one.
    top_model.load_weights(top_weights_path)
#    model.load_weights(top_weights_path)   #### change if 'm'

    # based_model_last_block_layer_number points to the layer in your model you want to train.
    # For example if you want to train the last block of a 19 layer VGG16 model this should be 15
    # If you want to train the last Two blocks of an Inception model it should be 172
    # layers before this number will used the pre-trained weights, layers above and including this number
    # will be re-trained based on the new data.
    for layer in top_model.layers[:based_model_last_block_layer_number]:
        layer.trainable = False
    for layer in top_model.layers[based_model_last_block_layer_number:]:
        layer.trainable = True

    model = ModelMGPU(top_model, gpus=4)

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    
    # save weights of best training epoch: monitor either val_loss or val_acc
    final_weights_path = os.path.join(os.path.abspath(model_path), 'model_weights_test.h5')
    callbacks_list = [
        ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=0, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        keras.callbacks.TensorBoard(log_dir='tensorboard/inception-resnet-v2-fine-tune', histogram_freq=0, write_graph=False, write_images=False)
    ]
    

    # fine-tune the model
    model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // batch_size,
                        epochs=nb_epoch,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        callbacks=callbacks_list)

    # save model
    model_json = top_model.to_json()
    with open(os.path.join(os.path.abspath(model_path), 'model_weights_test.json'), 'w') as json_file:
        json_file.write(model_json)


#train('data/train', 'data/validation', '.')

if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print('Arguments must match:\npython code/fine_tune.py <data_dir/> <model_dir/>')
        print('Example: python code/fine_tune.py data/dogs_cats/ model/dog_cats/')
        sys.exit(2)
    else:
        data_dir = os.path.abspath(sys.argv[1])
        train_dir = os.path.join(os.path.abspath(data_dir), 'train_dir')  # Inside, each class should have it's own folder
        validation_dir = os.path.join(os.path.abspath(data_dir), 'valid_dir')  # each class should have it's own folder
        model_dir = os.path.abspath(sys.argv[2])

        os.makedirs(os.path.join(os.path.abspath(data_dir), 'preview'))
        os.makedirs(model_dir)

    train(train_dir, validation_dir, model_dir)  # train model

    # release memory
    k.clear_session()
