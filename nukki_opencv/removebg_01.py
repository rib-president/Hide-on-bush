#-*- coding: utf-8 -*-

import cv2
import numpy as np
import os

'''
보통 
CANNY_THRESH_1 = 0
CANNY_THRESH_2 = 40

약하게
CANNY_THRESH_1 = 5
CANNY_THRESH_2 = 12~20

강하게 
CANNY_THRESH_1 = 5
CANNY_THRESH_2 = 100


'''
#== Parameters =======================================================================
BLUR = 21
CANNY_THRESH_1 =4
CANNY_THRESH_2 = 5
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # In BGR format


#== Processing =======================================================================

#-- Read image -----------------------------------------------------------------------
img_path = "../cropagainsave"
save_root_path = "../maskrcnn/"
folders_list = sorted(os.listdir(img_path))

def auto_canny(image, sigma=4.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	print('lower: %d  upper: %d' % (lower, upper))
 
	# return the edged image
	return edged

def filecheck(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print ('create' + path)
        except OSError as e:
            if e.errno != e.errno.EEXIST:
                raise

for folder_list in folders_list:
    folder_path = os.path.join(img_path,folder_list)
    barcodes_list = sorted(os.listdir(folder_path))
    for barcode_list in barcodes_list:
        barcode_path = os.path.join(folder_path,barcode_list)
        imgs_list = sorted(os.listdir(barcode_path))
        save_path = os.path.join(os.path.join(save_root_path , folder_list) , barcode_list)
        if os.path.exists(save_path):
            print(save_path + " exists")
            continue
        filecheck(save_path)
        for img_list in imgs_list:
            img = cv2.imread(os.path.join(barcode_path,img_list))
            print(os.path.join(barcode_path,img_list))
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            #-- Edge detection -------------------------------------------------------------------
            edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
            #edges = auto_canny(gray)
            edges = cv2.dilate(edges, None)
            edges = cv2.erode(edges, None)


            #-- Find contours in edges, sort by area ---------------------------------------------
            contour_info = []
            contours,hierachy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # Previously, for a previous version of cv2, this line was: 
            #  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # Thanks to notes from commenters, I've updated the code but left this note
            for c in contours:
                contour_info.append((
                    c,
                    cv2.isContourConvex(c),
                    cv2.contourArea(c),
                ))
            contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
            max_contour = contour_info[0]

            #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
            # Mask is black, polygon is white
            mask = np.zeros(edges.shape)
            cv2.fillConvexPoly(mask, max_contour[0], (255))

            #-- Smooth mask, then blur it --------------------------------------------------------
            mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
            mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
            mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
            mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

            #-- Blend masked img into MASK_COLOR background --------------------------------------
            mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
            img         = img.astype('float32') / 255.0                 #  for easy blending


            masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
            masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 
            # split image into channels
            c_red, c_green, c_blue = cv2.split(img)

            # merge with mask got on one of a previous steps
            img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))


            # save to disk
            cv2.imwrite(save_path+'/'+img_list.split('.')[0]+'.png', img_a*255)

