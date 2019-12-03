import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import colorsys
import cv2
import copy

ROOT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn import model as modellib
from mrcnn.visualize import display_images
from mrcnn.model import log
from keras import backend as K

import custom

WEIGHTS_PATH = "logs/mask_rcnn.h5"
MODEL_DIR = "./logs"
config = custom.CustomConfig()
DATASET_DIR = "video/"

class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

config = InferenceConfig()
DEVICE = "/gpu:0"
MODE = "inference"
RESULT_DIR = "result/"
MAX_COUNT = 1
mlp_data = []

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode=MODE, model_dir=MODEL_DIR, config=config)

model.load_weights(WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

def random_colors(N=20, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color_idx = random.randint(0, 19)
    color = colors[color_idx]
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
    return image

video_path = "video/"
video_folder_list = sorted(os.listdir(video_path))


for video_folder in video_folder_list :
    video_folder_path = os.path.join(video_path, video_folder)
    video_list = sorted(os.listdir(video_folder_path))

    for video in video_list:
        cap = cv2.VideoCapture(video_folder_path + '/' + video)
        #image = skimage.io.imread(video_path + video)
        print ('video name : ', video)
        cnt = 0  
        while True :
            ret, cv2img = cap.read()
            #cv2img = cv2.imread(video_path + video)
            print ('frame number : ', cnt)
            cnt += 1
            if ret == True:
                bboximg = copy.deepcopy(cv2img)
                results = model.detect([cv2img], verbose=0)
                r = results[0]
                
                bboxes = r['rois']
                scores = r['scores']
                masks = r['masks']
                n = bboxes.shape[0]

                #print(n)
                colors = random_colors()

                h, w, ch = cv2img.shape

                data = []
                avg = 0
    

                for i in range(n) :
                    bbox = bboxes[i]
                    xmin = bbox[1]
                    ymin = bbox[0]
                    xmax = bbox[3]
                    ymax = bbox[2]
    
                    print(xmin, ymin, xmax, ymax)

                    cv2.rectangle(bboximg, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
                    cv2.imshow('test', bboximg)
                    

                    cropped = cv2img[ymin:ymax, xmin:xmax]

                    cv2.imwrite('detect_crop/{2}_frame{0}_{1:05d}.jpg'.format(cnt,i,video), cropped)
                    #cv2.imwrite('result/' + img, masked)
                cv2.imwrite('detect_full/{1}_frame{0}.jpg'.format(cnt,video), bboximg)
                if cv2.waitKey(1) & 0xFF == ord('q') :
                    break


            else:
                break
        cap.release()

cap.release()
cv2.destroyAllWindows()
