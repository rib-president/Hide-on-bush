from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re
import os
import sys
import tarfile
import argparse
import datetime
import numpy as np
import tensorflow as tf
from six.moves import urllib
from numpy import array
from PIL import Image
import cv2
import io
import base64

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
model = "./inference/frozen_nasnet_mobile.pb"
model_graph = tf.Graph()
with model_graph.as_default():
    with tf.gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        input_layer = model_graph.get_tensor_by_name("input:0")
        output_layer = model_graph.get_tensor_by_name('final_layer/predictions:0')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
inference_session = tf.Session(graph = model_graph, config=config)


def decode_image(jpeg_file):
    with tf.device('/cpu:0'):
        decoder_graph = tf.Graph()
        with decoder_graph.as_default():
            decoded_image = tf.image.decode_jpeg(jpeg_file)
            normalized_image = tf.divide(decoded_image, 255)
            # reshaped_image = tf.reshape(normalized_image, [-1, 331, 331, 3])
        with tf.Session(graph = decoder_graph) as image_session:
        # image_session = tf.Session(graph = decoder_graph)
            input_0 = image_session.run(normalized_image)
    return input_0


def diagnose_image(inference_session, input_image):
    with tf.device('/gpu:0'):
        #print ("input_layer.shape, input_layer : ", input_layer.shape, input_layer)
        predictions = inference_session.run(output_layer, feed_dict={input_layer: input_image})
    predictions = np.squeeze(predictions)
    return predictions


def main(arguments):
    """ Inference the whole src root directory """
    src_root = "./demo/video"
    dst_root = "./demo/result"
    label_map_path = "./demo/labelmap/label.txt"
    if not os.path.isdir(dst_root):
        os.mkdir(dst_root)

    #images = os.listdir(src_root)
    #output_file = os.path.join(dst_root, "output_result.txt")
    #result_file = open(output_file, "a")

    label_map_file = open(label_map_path)
    label_map = {}
    for line_number, label in enumerate(label_map_file.readlines()):
        label_map[line_number] = label[:-1]
        line_number += 1
    label_map_file.close()


    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if ret:
    #for image in images:
    #    image_path = os.path.join(src_root, image)
    #    start = datetime.datetime.now()

            cv2_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_img, 'RGB')
            imgByteArr = io.BytesIO()
            imgByteArr.flush()
            pil_img.save(imgByteArr, format='JPEG')
            img_read = imgByteArr.getvalue()
            imgByteArr.flush()

            #with tf.gfile.FastGFile(imgByteArr, 'rb') as jpeg_file_raw:
            #    jpeg_file = jpeg_file_raw.read()
            #    input_0 = decode_image(jpeg_file)

            input_0 = decode_image(img_read)

            image_height = input_0.shape[0]
            image_width = input_0.shape[1]
            image_height_center = int(image_height/2)
            image_width_center = int(image_width/2)
    
            tl_crop = input_0[0:224, 0:224]
            tr_crop = input_0[0:224, image_width-224:image_width]
            bl_crop = input_0[image_height-224:image_height, 0:224]
            br_crop = input_0[image_height-224:image_height, image_width-224:image_width]
            center_crop = input_0[image_height_center - 112: image_height_center + 112, image_width_center - 112: image_width_center + 112]

            input_concat = np.asarray([tl_crop, tr_crop, bl_crop, br_crop, center_crop])
            input_batch = input_concat.reshape(-1, 224, 224, 3)
    
            predictions = diagnose_image(inference_session, input_batch)
            overall_result = np.argmax(np.sum(predictions, axis=0))

            cv2.imshow('detect', img)
            #result_file.write(image_path + "\n")
            #result_file.write(str(overall_result) + "\n")
     
            #end = datetime.datetime.now()
            print(label_map[overall_result])
            print (float(round(np.max(np.sum(predictions, axis=0))*10, 2)))
            #print("Time cost: ", end - start, "\n")

        #result_file.close()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))



cap.release()
cv2.destroyAllWindows()
