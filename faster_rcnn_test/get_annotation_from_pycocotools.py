#!/usr/bin/env python3

###benchmark
#    url = 'https://github.com/waspinator/pycococreator'
#    author = 'waspinator'
#    author_email = 'patrickwasp@gmail.com'

import datetime
import json
import os
import re
import fnmatch
import cv2
from PIL import Image
import numpy as np
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi


ROOT_DIR = 'train'
IMAGE_DIR = os.path.join(ROOT_DIR, "shapes_train")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations_train")

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'product',
        'supercategory': 'custom',
    },
]


def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info



def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def __get_annotation__(mask):

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    RLEs = cocomask.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
    RLE = cocomask.merge(RLEs)
    # RLE = cocomask.encode(np.asfortranarray(mask))
    area = cocomask.area(RLE)
    [x, y, w, h] = cv2.boundingRect(mask)

    return segmentation, [x, y, w, h], area



def main():

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    
    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            image = Image.open(image_filename)
            image_info = create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    
                    print(annotation_filename)
                    class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    mask = np.array(Image.open(annotation_filename).convert('L'))
                    #binary_mask = np.asarray(Image.open(annotation_filename)
                    #    .convert('1')).astype(np.uint8)
                    if np.all(mask == 0):
                        continue
                    
                    segmentation, bbox, area = __get_annotation__(mask)

                    annotation_info = {"segmentation" : segmentation,
                                        "area" : np.float(area),
                                        "iscrowd" : 0,
                                        "image_id" : image_id,
                                        "bbox" : bbox,
                                        "category_id" : category_info["id"],
                                        "id": segmentation_id}

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open('{}/instances_shape_custom_train.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
