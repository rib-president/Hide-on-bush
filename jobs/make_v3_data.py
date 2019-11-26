import os
import sys
import argparse
import numpy as np
import json
#import requests
import datetime as dt
import random
import collections
import cv2
import copy
import logging
from PIL import Image, ImageDraw, ImageFont
from threading import Thread, Lock
import CaffeDetection

# Make sure that caffe is on the python path:
caffe_root = '../caffe/'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

##change model
weights_name = 'number_SSD_300x300_iter_296091.caffemodel'
detection = CaffeDetection.Detection(weights_name)



def detect_main(img, save_path, count):
    result = detection.detect(img)

    img2 = copy.deepcopy(img)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(img2, 'RGB')
    for item in result:
        xmin = int(round(item[0] * width))
        ymin = int(round(item[1] * height))
        xmax = int(round(item[2] * width))
        ymax = int(round(item[3] * height))


        crop_img = img2.crop((xmin, ymin, xmax, ymax))
        crop_img.save(save_path + '/%05d.jpg' % count, 'JPEG')
        count += 1
    return count

        
        
        #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
        #text = item[6] + ':' + str(round(item[5], 2))
        #font = cv2.FONT_HERSHEY_PLAIN
        #font_scale = 2
        #thickness = 2
        #size = cv2.getTextSize(text, font, font_scale, thickness)
        #label_w, label_h = size[0]
        #line_type = 8
        #margin = 10
        #bottomLeftOrigin = False

        #cv2.rectangle(img, (xmin, ymin-10-label_h-margin), (xmin+label_w, ymin-10+label_h), (255, 255, 255), -1)
        #cv2.putText(img, text, (xmin, ymin-10), font, font_scale, (0, 0, 0), thickness, line_type, bottomLeftOrigin)

##change url
#cap = cv2.VideoCapture("rtsp://admin:emart1234@192.168.1.87:554/cam/realmonitor?channel=1&subtype=0")

src = './tv/5_11/'
dst = './crop/a'
video_list = sorted(os.listdir(src))

for video in video_list:
    count = 0
    video_path = os.path.join(src, video)
    save_path = os.path.join(dst, video[:-4])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cap = cv2.VideoCapture(video_path)
    #cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, 1)

    while True:
        ret, img = cap.read()

        if ret:
            height, width = img.shape[:2]
            img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
            count = detect_main(img, save_path, count)

            print count
    #        img = detect_main(img)

            #if ret == True:
            #    cv2.imshow('frame',img)
            #    if cv2.waitKey(1) & 0xFF == ord('q'):
            #        break
            #else:
            #    break
            
        else:
            break

#cap.release()
#cv2.destroyAllWindows()


