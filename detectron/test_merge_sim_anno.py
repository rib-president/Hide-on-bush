from PIL import Image, ImageFilter, ImageEnhance
from Transforms import RGBTransform
import math
import numpy as np
import random
import diagonal_crop
import os
import cv2
import xml.etree.cElementTree as ET
import sys
import copy
import json
import datetime
import re
import fnmatch
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi



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
        'supercategory': 'nobrand',
    },
]








class Simulator():
    def __init__(self, img_num, png_path, back_path, width, height, saving_path):
        self.img_num = img_num
        self.png_path = png_path
        self.back_path = back_path
        self.max_object_num = 10

        self.img_path = saving_path
        self.contour_path = './contour/'
        self.check_folder_exists(self.img_path)

        self.jsons = []

        '''
        object filter list
        0, 1, 2: RGB
        3, 4, 5: gradient
        6, 7, 8: brightness, contrast, blur
        9: slice
        10: motion blur

        background filter list
        0, 1, 2: RGB
        3, 4: brightness, contrast
        5: translate
        6: rotate and crop
        7: blur

        '''

        self.img_w = int(width)
        self.img_h = int(height)

        self.filter_list = [0, 1, 2, 6, 7]
        self.bg_filter_list = [3, 4, 5, 6]


    def create_data(self):
        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }
        image_id = 1
        segmentation_id = 1
        category_id = 1    
    
        png_dirs = os.listdir(self.png_path)
        bg_list = os.listdir(self.back_path)

        num_png_dir = len(png_dirs)

        for i in range(self.img_num):
            #self.check_folder_exists(self.img_path + png_dir)
            cropped_img_data = []
            x_coords = []
            y_coords = []
            contour_x = []
            contour_y = []
            categories = []
            attached = []

            max_num_per_img = random.randint(1, self.max_object_num)
            #max_num_per_img = 1
            for object_num in range(max_num_per_img):
                self.ii = 0
                png_dir = random.choice(png_dirs)
                png = random.choice(os.listdir(self.png_path + '/' + png_dir))
                #augmented_img = self.augment_default(self.png_path + '/' + png_dir + '/' + png)
                #png = random.choice(os.listdir(self.png_path))
                augmented_img = self.augment_default(self.png_path + '/' + png_dir + '/' + png)

                applying_filter = random.randint(0, 1)
                if applying_filter:
                    filters = random.sample(set(self.filter_list), 2)
                    print('apply filter', filters)
                    if 9 in filters:
                        sliced_img = self.slice_image(augmented_img)
                        sw, sh = sliced_img.size
                        filter_not_slice = [f for f in filters if f != 12]
                        if sw != 0 and sh != 0:
                            filter_img = self.augment(filter_not_slice[0], sliced_img)
                        else:
                            # case of slice error
                            filter_img = self.augment(filter_not_slice[0], augmented_img)
                    else:
                        filter_img = self.augment(filters[0], augmented_img)
                        filter_img = self.augment(filters[1], filter_img)

                    if filter_img.size == (0, 0):
                        filter_img = augmented_img

                else:
                    print('no filter')
                    filter_img = augmented_img

                w, h = filter_img.size
                x = random.randint(0, 1250 - w)
                y = random.randint(0, self.img_h - h)
                cropped_img_data.append(filter_img)
                x_coords.append(x)
                y_coords.append(y)
                categories.append(1)
                attached.append(1)
               

            bg = random.choice(bg_list)
            bg_img = Image.open(self.back_path + bg)

            bg_applying_filter = random.randint(0, 1)
            print(bg_applying_filter)
            if bg_applying_filter:
                bg_filters = random.sample(set(self.bg_filter_list), 2)
                bg_filtered_img = self.augment_background(bg_filters[0], bg_img)
                bg_filtered_img = self.augment_background(bg_filters[1], bg_img)
            else:
                bg_filtered_img = bg_img

            for a in range(0, max_num_per_img - 1):
                for b in range(a+1, max_num_per_img):
                    intersection = self.check_intersection(cropped_img_data[a], cropped_img_data[b], x_coords[a], y_coords[a], x_coords[b], y_coords[b])
                    if intersection > 0:
                        cropped_w, cropped_h = cropped_img_data[a].size
                        # if two object image intersect more 30%, set attatchCheck '0'. 
                        if (intersection > cropped_w * cropped_h * 0.5):
                            attached[a] = 0

            for idx in range(len(cropped_img_data)):
                if attached[idx] == 1:
                    bg_img.paste(cropped_img_data[idx], (x_coords[idx], y_coords[idx]), cropped_img_data[idx])
                    white_back = Image.new('RGB', bg_img.size, (255, 255, 255))
                    black_img = Image.new('RGBA', cropped_img_data[idx].size, (0, 0, 0))
                    white_back.paste(black_img, (x_coords[idx], y_coords[idx]), mask=cropped_img_data[idx])
                    maskimg = cv2.cvtColor(np.array(white_back), cv2.COLOR_RGB2BGR)

                    segmentation, bbox, area = self.contour(maskimg, i)

                    annotation_info = {"segmentation" : segmentation,
                        "area" : np.float(area),
                        "iscrowd" : 0,
                        "image_id" : image_id,
                        "bbox" : bbox,
                        "category_id" : category_id,
                        "id": segmentation_id}
                        
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
                    segmentation_id += 1
                    '''
                    if len(all_x) is not 0:
                        contour_x.append(all_x[0])
                        contour_y.append(all_y[0])
                        print('find %d contour' % len(all_x))
                        print('class: %s' % categories[idx])
                    else:
                        print('***** NO CONTOUR!!!!! *****')
                    '''
                    #cv2.imwrite('{}/{}_prouct_{}.jpg'.format(self.contour_path, i, idx), maskimg)
            image_info = create_image_info(
                image_id, i, bg_img.size)
            coco_output["images"].append(image_info)
            bg_img.save('%s/%05d.jpg' % (self.img_path, i), 'JPEG')
            #self.create_json_annotation(i, max_num_per_img, categories, contour_x, contour_y)
            print('image %s/%05d.jpg saved.' % (self.img_path, i))
            image_id += 1
            
        with open('{}/instances_shape_coco_train.json'.format(ROOT_DIR), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)
        #self.write_json()
        #self.jsons = []

    # Check Folder. If the folder is not exist create folder
    def check_folder_exists(self, path):
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                print 'create ' + path
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    # default augmented options
    def augment_default(self, path):
	img = Image.open(path)
	# rotated object
	rotated_image = self.rotate_image(img)
	# recroped object for getting exact object image
        while (True):
            resized_image = self.resize_image(rotated_image)
            recropped_image = self.recrop_image(resized_image)
            if self.check_object_size(recropped_image) == True:
                break
            else:
                rotated_image = recropped_image
	return recropped_image

    # Check object image size 
    def check_object_size(self, image):
        w, h = image.size
        print(w, h)
        if w < self.img_w and h < self.img_h:
            return True

    ## rotate Image 
    def rotate_image(self, image):
        img = image
        rotatednum = random.randint(-90, 90) # choice rotation angle randomly
        rotatedImg = img.rotate(rotatednum, expand=True) # roate image
        return rotatedImg

    # resize image 
    def resize_image(self, image):
        img = image
        ratio = random.uniform(0.3, 0.5) # choice ratio randomly
        width, height = img.size
        # set resize image size
        size = int(round(width*ratio, 0)), int(round(height*ratio, 0))
        resizeImg = img.resize(size, Image.ANTIALIAS) # resize image
        return resizeImg

    # crop image for exact image
    # if just crop image, the image is not exact crop(they have blank value). 
    # so, need again crop image exactly
    def recrop_image(self, image):
        img = image.convert('RGBA')
        pixeldata = list(img.getdata())
        width, height = img.size
        x_list = []
        y_list = []
        for i, pixel in enumerate(pixeldata):
            if pixel[3:4] == (255,):
                y, x = divmod(i, width)
                x_list.append(x)
                y_list.append(y)
        if len(x_list) > 0 or len(y_list) > 0:
            maxX = max(x_list)
            maxY = max(y_list)
            minX = min(x_list)
            minY = min(y_list)
            recropped_img = img.crop((minX, minY, maxX, maxY))
        else:
            recropped_img = img.crop((0, 0, 0, 0))
        return recropped_img

    # get Alpha value of object image
    def get_alpha(self, image):
        img = image.convert('RGBA')
        pixeldata = list(img.getdata())
        count = 0
        # if each pixel have alpha value, count.
        for i, pixel in enumerate(pixeldata):
            if pixel[3:4] == (255,):
                count += 1
        return count


    def augment(self, filter_num, image):
        output_img = image
        if filter_num == 0:
            output_img = self.balanceR(image, 2)
        elif filter_num == 1:
            output_img = self.balanceG(image, 2)
        elif filter_num == 2:
            output_img = self.balanceB(image, 2)
        elif filter_num == 3:
            output_img = self.gradient(image, 0)
        elif filter_num == 4:
            output_img = self.gradient(image, 1)
        elif filter_num == 5:
            output_img = self.gradient(image, 2)
        elif filter_num == 6:
            output_img = self.brightness(image)
        elif filter_num == 7:
            output_img = self.contrast(image)
        elif filter_num == 8:
            output_img = self.blur(image)
        elif filter_num == 9:
            output_img = self.slice_image(image)
        elif filter_num == 10:
            output_img = self.motionblur(image)
        return output_img


    def augment_background(self, filter_num, image):
        output_img = image
        if filter_num == 0:
            output_img = self.balance_background_rgb(image, 0)
        elif filter_num == 1:
            output_img = self.balance_background_rgb(image, 1)
        elif filter_num == 2:
            output_img = self.balance_background_rgb(image, 2)
        elif filter_num == 3:
            output_img = self.contrast_background(image)
        elif filter_num == 4:
            output_img = self.brightness_background(image)
        elif filter_num == 5:
            output_img = self.translate_background(image)
        elif filter_num == 6:
            output_img = self.rotate_crop_background(image)
        elif filter_num == 7:
            output_img = self.blur_background(image)
        return output_img

    # r balance
    def balanceR(self, image, gradNum):
        input_im = image
        if gradNum == 0: # dark	part
            num = round(random.uniform(0.2, 1.0), 1)
            output_im = RGBTransform().mix_with((255, 0, 0), factor=num).applied_to(input_im)
        elif gradNum == 1: # bright part
            num = round(random.uniform(0.05, 0.2), 2)
            output_im = RGBTransform().mix_with((255, 0, 0), factor=num).applied_to(input_im)
        else:
            num = round(random.uniform(0.05, 0.2), 2)
            output_im = RGBTransform().mix_with((255, 0, 0), factor=num).applied_to(input_im)
        return output_im

    # g balance
    def balanceG(self, image, gradNum):
        input_im = image
        if gradNum == 0: # dark	part
            num = round(random.uniform(0.2, 1.0), 1)
            output_im = RGBTransform().mix_with((0, 255, 0), factor=num).applied_to(input_im)
        elif gradNum == 1: # bright part
            num = round(random.uniform(0.05, 0.2), 2)
            output_im = RGBTransform().mix_with((0, 255, 0), factor=num).applied_to(input_im)
        else:
            num = round(random.uniform(0.05, 0.2), 2)
            output_im = RGBTransform().mix_with((0, 255, 0), factor=num).applied_to(input_im)
        return output_im


    # b balance
    def balanceB(self, image, gradNum):
        input_im = image
        if gradNum == 0: # dark	part
            num = round(random.uniform(0.2, 1.0), 1)
            output_im = RGBTransform().mix_with((0, 0, 255), factor=num).applied_to(input_im)
        elif gradNum == 1: # bright part
            num = round(random.uniform(0.05, 0.2), 2)
            output_im = RGBTransform().mix_with((0, 0, 255), factor=num).applied_to(input_im)
        else:
            num = round(random.uniform(0.05, 0.2), 2)
            output_im = RGBTransform().mix_with((0, 0, 255), factor=num).applied_to(input_im)
        return output_im


    def gradient(self, image, gradnum):
        input_im = image
        if input_im.mode != 'RGBA':
            input_im = input_im.convert('RGBA')
        width, height = input_im.size

        # brightness of dark part
        rgbNum = gradnum
        gradNum = 0 # dark part
        dark_im = self.augment_gradient(rgbNum, input_im, gradNum)
        dark_im = self.brightnessGrad(dark_im, 0)

        alpha_gradient = Image.new('L', (1, height), color='black')

        gradient=1.5
        initial_opacity=1

        for y in range(height):
            a = int((initial_opacity * 255.) * (1. - gradient * float(y)/height))
            if a > 0:
                alpha_gradient.putpixel((0, y), a)
            else:
                alpha_gradient.putpixel((0, y), 0)

		alpha = alpha_gradient.resize(dark_im.size)

        # brightness for bright park
        gradNum = 1 # bright part
        bright_im = self.augment_gradient(rgbNum, input_im, gradNum)
        bright_im = self.brightnessGrad(bright_im, 1)
        bright_im.putalpha(alpha)

        output_im = Image.alpha_composite(dark_im, bright_im)
        output_im = output_im.convert('RGBA')

        maskedImg = Image.new('RGBA', image.size)
        maskimg = image
        maskedImg.paste(output_im, (0, 0), mask=maskimg)

        ## rotate image for 360 gradient
        rotatedImg = self.rotate_image(maskedImg)
        rocheck = True
        while(rocheck):
            rotatedImg = self.recrop_image(rotatedImg)
            roWidth, roHeight = rotatedImg.size
            if roHeight < self.img_h:
                rocheck = False
        return rotatedImg


    def augment_gradient(self, num, image, gradNum):
        if num==0: #balanceR
            output_im = self.balanceR(image, gradNum)
        elif num==1: #balanceG
            output_im = self.balanceG(image, gradNum)
        elif num==2: # balanceB
            output_im = self.balanceB(image, gradNum)
        return output_im


    # brightenss for gradient dark and bright part
    def brightnessGrad(self, image, num):
        if num == 0: #dark part
            input_im = image
            brightness = round(random.uniform(0.1, 0.9), 1)
            enhancer = ImageEnhance.Brightness(input_im)
            output_im = enhancer.enhance(brightness)
        elif num == 1: #bright part
            input_im = image
            brightness = round(random.uniform(1.0, 2.0), 1)
            enhancer = ImageEnhance.Brightness(input_im)
            output_im = enhancer.enhance(brightness)
        return output_im



    # brightness filter for object image
    def brightness(self, image):
        input_im = image
        num = round(random.uniform(0.3, 2), 1)
        bright = ImageEnhance.Brightness(input_im)
        output_im = bright.enhance(num)
        return output_im

    # contrast filter for object image
    def contrast(self, image):
        img = image
        num = round(random.uniform(0.1, 1.5), 1)
        enhancer = ImageEnhance.Contrast(img)
        cont = enhancer.enhance(num)
        return cont

    # blur filter for object image
    def blur(self, image):
        input_im = image
        num = round(random.uniform(2.0, 3.0), 1)
        output_im = input_im.filter(ImageFilter.GaussianBlur(num))
        return output_im

    # motionblur fiter for object image
    def motionblur(self, image):
        pil_image = image
        img = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGBA2BGRA)
        size = random.randint(1, 50)

        # generating the kernel
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)

        #print kernel_motion_blur
        kernel_motion_blur = kernel_motion_blur / size

        # applying the kernel to the input image
        motionblurImg = cv2.filter2D(img, -1, kernel_motion_blur)
        cv2_im = cv2.cvtColor(motionblurImg, cv2.COLOR_BGRA2RGBA)
        output_im = Image.fromarray(cv2_im)
        return output_im

    ## crop(slice) Image 
    # cnum - cnum == 0 : for check object image. return croped image and check
    # cnum == 1 : return only croped image
    # reference site : https://github.com/jobevers/diagonal-crop
    # If need more explaination of crop filter, read Document named '2_SSD Data Generator'
    def slice_image(self, image):#cnum):
        img = image
        width, height = img.size
        rdiagonal = math.sqrt(pow(width, 2) + pow(height, 2))
        check = True

        ## rotate diagonal square randomly. diagonal square means cropped image shape
        randnum = random.randint(1, 4)
        if randnum == 1: # x = 0, y > 0
            baseX = 0
            baseY = random.randint(0, height)
            angle = random.randint(0, 90)
        elif randnum == 2: # x > 0, y = 0
            baseX = random.randint(0, width)
            baseY = 0
            angle = random.randint(90, 180)
        elif randnum == 3: # x = width, y > 0
            baseX = width
            baseY = random.randint(0, height)
            angle = random.randint(180, 270)
        elif randnum == 4: # x > 0, y = height
            baseX = random.randint(0, width)
            baseY = height
            angle = random.randint(270, 360)

        base = (baseX, baseY) # diagonal square start point
        h = random.randint(int(width * 0.3), width) # diagonal square height
        w = rdiagonal # diagonal square width 
        cropped = diagonal_crop.crop(img, base, angle, h, w) # crop object image
        recropped = self.recrop_image(cropped) # crop object image exactly
        rewidth, reheight = recropped.size

        ## check croped object image size
        # 1. croped object image's size is bigger than 40% of original object image area.
        # 2. croped object image's alpha values are bigger than 30% of croped image's area,
        # because some object images have not full alpha values.
        if reheight < 900:
           if rewidth < 540:
               # (1.) bigger than 40% of original object image area.
               if rewidth * reheight > width * height * 0.4:
                   # (2.) get alpha values for checking object image's area
                   count = self.get_alpha(recropped)
                   if count > int(rewidth * reheight * 0.3):
                       check = False
        '''
        if cnum == 0: # for check object image. return croped image and check
            return (cropped, check)
        elif cnum == 1: # return only croped image
            return cropped
        '''
        return recropped


    def balance_background_rgb(self, image, gradNum):
        input_im = image
        if gradNum == 0 : # balance R
            num = round(random.uniform(0.01, 0.09), 2)
            output_im = RGBTransform().mix_with((255, 0, 0), factor=num).applied_to(input_im)
        elif gradNum == 1 : # balance G
            num = round(random.uniform(0.01, 0.09), 2)
            output_im = RGBTransform().mix_with((0, 255, 0), factor=num).applied_to(input_im)
        elif gradNum == 2 : # balance B
            num = round(random.uniform(0.01, 0.09), 2)
            output_im = RGBTransform().mix_with((0, 0, 255), factor=num).applied_to(input_im)
        return output_im


    def contrast_background(self, image):
        num = round(random.uniform(0.3, 1.5), 1)
        enhancer = ImageEnhance.Contrast(image)
        augmented = enhancer.enhance(num)
        return augmented


    def brightness_background(self, image):
        img = image
        num = round(random.uniform(0.5, 2.5), 1)
        enhancer = ImageEnhance.Brightness(image)
        augmented = enhancer.enhance(num)
        return augmented


    def blur_background(self, image):
        num = round(random.uniform(1.0, 2.0), 1)
        augmented = image.filter(ImageFilter.GaussianBlur(num))
        return augmented


    def translate_background(self, image):
        resized = image.resize((1920, 1080))
        x = random.randint(0, 640)
        y = random.randint(0, 360)
        w = self.img_w
        h = self.img_h
        augmented = resized.crop((x, y, x+w, y+h))
        return augmented


    def rotate_background(self, image, angle=None):
        if angle == None:
            rotated_num = random.randint(-5, 5)
            rotated_img = image.rotate(rotated_num, expand=True)
        else:
            rotated_img = image.rotate(angle, expand=True)
        return (rotated_img, rotated_num)


    def rotate_crop_background(self, image):
        width, height = image.size
        rotated, rotated_num = self.rotate_background(image)
        aw, ah = rotated.size

        if not rotated_num == 0:
            if rotated_num > 0: # angle >0
                a = ah - height * math.sin(90 - (math.pi * rotated_num / 180))
                b = aw - width * math.cos(math.pi * rotated_num / 180)
            elif rotated_num < 0: # angle < 0
                a = ah - height * math.cos(-math.pi * rotated_num / 180)
                b = aw - width * math.sin(90 + math.pi * rotated_num/180)

            cropped = rotated.crop((int(b), int(a), int(aw-b), int(ah-a)))
            cropped = cropped.resize((self.img_w, self.img_h))

        else:
            cropped = rotated
            cropped = cropped.resize((self.img_w, self.img_h))

        return cropped


    # Check Intersection of object images
    def check_intersection(self, im1, im2, im1X, im1Y, im2X, im2Y):
        im1W, im1H = im1.size
        im2W, im2H = im2.size
        im1Left = im1X
        im1Right = im1X + im1W
        im1Top = im1Y
        im1Bottom = im1Y + im1H
        im2Left = im2X
        im2Right = im2X + im2W
        im2Top = im2Y
        im2Bottom = im2Y + im2H
        dx = min(im1Right, im2Right) - max(im1Left, im2Left)
        dy = min(im1Bottom, im2Bottom) - max(im1Top, im2Top)
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        else:
            return 0


    def write_txt(self, filenum, imgs, x_coords, y_coords, category, attached, img_size):
        with open('%s%05d.txt' % (self.label_path, filenum), 'w') as f:
            for i in range(len(imgs)):
                if attached[i] == 1:
                    w, h, xmin, ymin, xmax, ymax, xc, yc = self.normalize(imgs[i].size, x_coords[i], y_coords[i], img_size)
                    # format: filename, label, w, h, xmin, ymin, xc, yc
                    f.write('%05d.jpg,%s,%f,%f,%f,%f,%f,%f\n' % (i, category[i], w, h, xmin, ymin, xc, yc))

            print('%s%05d.txt saved.' % (self.label_path, filenum))


    def write_xml(self, filenum, imgs, x_coords, y_coords, category, attached, img_size):
        root = ET.Element("annotation")
        foldername = ET.SubElement(root, "folder")
        foldername.text = "tracking"

        filename = ET.SubElement(root, "filename")
        fname = '%05d.jpg' % filenum
        filename.text = fname

        size = ET.SubElement(root, "size")
        w = ET.SubElement(size, "width")
        h = ET.SubElement(size, "height")
        depth = ET.SubElement(size, "depth")
        w.text = str(img_size[0])
        h.text = str(img_size[1])
        depth.text = '3'

        segmented = ET.SubElement(root, "segmented")
        segmented.text = '0'

        for i in range(len(imgs)):
            if attached[i] == 1:
                objectn = ET.SubElement(root, "object")
                objw, objh = imgs[i].size

                name = ET.SubElement(objectn, "name")
                name.text = 'product'

                pose = ET.SubElement(objectn, "pose")
                pose.text = 'Unspecified'

                truncated = ET.SubElement(objectn, "truncated")
                truncated.text = '0'

                difficult = ET.SubElement(objectn, "difficult")
                difficult.text = '0'

                bndbox = ET.SubElement(objectn, "bndbox")
                xmin = ET.SubElement(bndbox, "xmin")
                xmin.text = str(x_coords[i])
                ymin = ET.SubElement(bndbox, "ymin")
                ymin.text = str(y_coords[i])
                xmax = ET.SubElement(bndbox, "xmax")
                xmax.text = str(x_coords[i] + objw)
                ymax = ET.SubElement(bndbox, "ymax")
                ymax.text = str(y_coords[i] + objh)

            tree = ET.ElementTree(root)
            tree.write('%s%05d.xml' % (self.xml_path, filenum))
            print('%s%05d.xml saved.' % (self.xml_path, filenum))


    def normalize(self, obj_size, x, y, img_size):
        ow, oh = obj_size
        w, h = img_size
        xmin = x
        ymin = y
        xmax = x + ow
        ymax = y + oh
        xc = (ow / 2) + xmin
        yc = (oh / 2) + ymin

        fow = float(ow) / w
        foh = float(oh) / h
        fxmin = float(xmin) / w
        fymin = float(ymin) / h
        fxmax = float(xmax) / w
        fymax = float(ymax) / h
        fxc = float(xc) / w
        fyc = float(yc) / w

        return (fow, foh, fxmin, fymin, fxmax, fymax, fxc, fyc)


    def contour(self, maskimg, filename):
        all_x = []
        all_y = []
        xpoints = []
        ypoints = []
        img = maskimg
        #img = cv2.resize(img, None,fx=0.25, fy=0.25, interpolation = cv2.INTER_CUBIC)

        # get a blank canvas for drawing contour on and convert img to grayscale
        canvas = np.zeros(img.shape, np.uint8)
        img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # filter out small lines between counties
        kernel = np.ones((5,5),np.float32) / 25
        img2gray = cv2.filter2D(img2gray,-1,kernel)

        # threshold the image and extract contours
        ret,thresh = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY_INV)
        # opencv 3.x
        contour_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # opoencv 2.x
        #contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.imwrite('{0}/{1:05d}_product_{2}.png'.format(self.contour_path, filename, self.ii), contour_img)
        #count = 0
        
        segmentation, bbox, area = __get_annotation__(contour_img)
        '''
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            hull = cv2.convexHull(cnt)
            ct_img = cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 3)
            #cv2.imwrite('contour/%05d_%d.jpg' % (filename, count), ct_img)

            for idx, coords in enumerate(approx):
                xpoints.append(coords[0][0])
                ypoints.append(coords[0][1])

            all_x.append(xpoints)
            all_y.append(ypoints)

            xpoints = []
            ypoints = []
            count += 1
        self.ii += 1
        '''
        return segmentation, bbox, area


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



    def create_json_annotation(self, filecount, max_num, category, contour_x, contour_y):
        data = {}
        fileinfo = {}
        file_attr = {}
        regions = {}

        for i in range(0, len(category)):
            shape_attr = {}
            region_attr = {}
            region_id = {}

            regions[str(i)] = region_id

            region_id["shape_attributes"] = shape_attr
            region_id["region_attributes"] = region_attr

            x_coords_list = np.array(contour_x[i], np.int32)
            x_coords_list = x_coords_list.tolist()
            y_coords_list = np.array(contour_y[i], np.int32)
            y_coords_list = y_coords_list.tolist()

            shape_attr["name"] = "polygon"
            shape_attr["all_points_x"] = x_coords_list
            shape_attr["all_points_y"] = y_coords_list

        file_attr["fileref"] = ""
        file_attr["size"] = os.path.getsize(self.img_path + '%05d.jpg' % filecount)
        file_attr["filename"] = '%05d.jpg' % filecount
        file_attr["base64_img_data"] = ""
        file_attr["file_attributes"] = {}
        file_attr["regions"] = regions

        self.jsons.append(file_attr)

    def write_json(self):
        imglist = sorted(os.listdir(self.img_path))
        data = {}
        for idx, imgfile in enumerate(imglist):
            #print 'self.jsons idx = ', self.jsons[idx]
            data['{0}'.format(imgfile)] = self.jsons[idx]

            with open(self.img_path + '/via_region_data.json', 'w') as f:
                f.write(json.dumps(data))

if __name__ == '__main__':
    final_num = int(sys.argv[1])
    folder_path = sys.argv[2]
    back_path = sys.argv[3]
    width = sys.argv[4]
    height = sys.argv[5]
    saving_path = sys.argv[6]
    simulator = Simulator(final_num, folder_path, back_path, width, height, saving_path)
    simulator.create_data()
