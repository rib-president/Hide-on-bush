import sys
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

caffe_root = '/home/rib/caffe/'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2

class Detection:
    def __init__(self, weights_name):
        self.gpu_id = 0
        self.ssd_root = '/home/rib/ssd/'
        self.model_def = self.ssd_root + 'models/SSD_300x300/deploy.prototxt'
        self.model_weights = self.ssd_root + 'models/snapshot/SSD_300x300/' + weights_name
        self.image_resize = 300
        self.labelmap_file = self.ssd_root + 'dataset/labelmap.prototxt'

        caffe.set_device(self.gpu_id)
        caffe.set_mode_gpu()

        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(self.model_def,      # defines the structure of the model
                             self.model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(self.labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def get_labelname(self, labelmap, labels):
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

    def detect(self, image, conf_thresh=0.3, topn=20):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        #image = caffe.io.load_image(image_file)

        #Run the net and examine the top_k results
        #transformed_image = self.transformer.preprocess('data', image)
        resized = cv2.resize(image, (300, 300))
        #cv2.imwrite(ssd_root + 'test.jpg', resized)
        npimg = np.array(resized)
        reshaped = npimg.transpose(2, 0, 1)
        exp_img = np.expand_dims(reshaped, axis=0)
        self.net.blobs['data'].data[...] = exp_img

        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = self.get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result
