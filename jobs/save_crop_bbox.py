import os
import sys
import cv2
import copy
from PIL import Image
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



def detect_main(img, count):
    result = detection.detect(img)

    img2 = copy.deepcopy(img)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = Image.fromarray(img2, 'RGB')
    

    
    for i in range(len(result)):
        xmin = int(round(result[i][0] * width))
        ymin = int(round(result[i][1] * height))
        xmax = int(round(result[i][2] * width))
        ymax = int(round(result[i][3] * height))

        text = result[i][6] + ',' + str(round(result[i][5]))
        crop_img = img2.crop((xmin, ymin, xmax, ymax))

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)
        cv2.putText(img,text , (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 2, 5, False)
        crop_img.save(save_path + '/%05d.jpg' % count, 'JPEG')
        count += 1
        
    cv2.imwrite(save_path_2 + '/%05d.jpg' % count, img)
    return count

        

src = './zzz/'
dst = './crop'
dst_2 = './bbox'

folder_list = sorted(os.listdir(src))

for folder in folder_list:
    folder_path = os.path.join(src, folder)
    save_path = os.path.join(dst, folder)
    save_path_2 = os.path.join(dst_2, folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_2):
        os.makedirs(save_path_2)    
    img_list = sorted(os.listdir(folder_path))
    count = 0
    for img in img_list:
        img_path = os.path.join(folder_path, img)
        image = cv2.imread(img_path)
        height, width = image.shape[:2]
        count = detect_main(image, count)

        print count

