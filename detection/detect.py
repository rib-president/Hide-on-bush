#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw 
# Make sure that caffe is on the python path:
caffe_root = '/home/rib/caffe'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

import cv2
import copy
from CaffeDetection import Detection


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def main():
    '''main '''
    detection = Detection('number_SSD_300x300_iter_120000.caffemodel')

    width = 1280
    height = 720

    imglist = sorted(os.listdir('detecting_img/'))
    for count, imgfile in enumerate(imglist):
        img = cv2.imread('detecting_img/' + imgfile)
        result = detection.detect(img)
        print result

        for item in result:
            xmin = int(round(item[0] * width))
            ymin = int(round(item[1] * height))
            xmax = int(round(item[2] * width))
            ymax = int(round(item[3] * height))

            text = '%s %f' % (item[6], item[5])

            font = cv2.FONT_HERSHEY_PLAIN
            font_scale = 1.5
            thickness = 2
            line_type = 8
            margin = 10

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
            cv2.putText(img, text, (xmin, ymin - 10), font, font_scale, (0, 0, 0), thickness, line_type, False)

        #img.save('detect_result.jpg')
        cv2.imwrite('result/%05d.jpg' % count, img)
        cv2.imshow('detected', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
