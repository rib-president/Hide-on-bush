#-*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import copy
from PIL import Image

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

r=63
drawing = False # true if mouse is pressed
mode = True # if True, erase image. Press 'm' to toggle to add image
ix,iy = -1,-1
#== Processing =======================================================================

#-- Read image -----------------------------------------------------------------------
img_path = "./image"
save_root_path = "./image_result"
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


# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode, image_new, img_a
    cnt = 0

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                img_a = np.array(img_a)#.astype('float32')
                cv2.circle(numpy_vertical,(x,y),r,(0,200,0),-1)
                cv2.circle(img_a,(x-img_a.shape[1],y),r,(0,200,0),-1)
                #image_new = cv2.addWeighted(img_a*255, 1, img3, 0, 0)
            else:
                try:
                    img_a = Image.fromarray((img_a).astype(np.uint8))
                except:
                    image_new, img_a = copy_bg_to_fg(x,y, img_a)
                    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.circle(numpy_vertical,(x,y),r,(0,200,0),-1)
            cv2.circle(img_a,(x-img_a.shape[1],y),r,(0,200,0),-1)        
            image_new = cv2.addWeighted(img_a, 1, img3, 0, 0)

        else:
            try:
                img_a = Image.fromarray((img_a).astype(np.uint8))
            except:
                image_new, img_a = copy_bg_to_fg(x,y, img_a)
                

            



def copy_bg_to_fg(x,y, img_a):
    cv2.circle(numpy_vertical,(x,y),r,(255,0,0),-1)
    mask_img = img2[y-r:y+r, x-r:x+r]
    rgba = cv2.cvtColor(mask_img, cv2.COLOR_RGB2RGBA)
    mask = np.full((mask_img.shape[0], mask_img.shape[1]), 0, dtype=np.uint8)
    cv2.circle(mask,(r,r), r, (255,255,255),-1)
    fg = cv2.bitwise_or(rgba, rgba, mask=mask)                 
    add_img = Image.fromarray(fg)
    #img_a = Image.fromarray((img_a*255).astype(np.uint8))
    img_a.paste(add_img, (x-r, y-r), add_img)
    #img_a = np.array(img_a)
    image_new = img_a
    return image_new, img_a



def remove_back(img):
    #img = image.convert('RGBA')
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



for folder_list in folders_list:
    folder_path = os.path.join(img_path,folder_list)
    imgs_list = sorted(os.listdir(folder_path))
    save_path = os.path.join(save_root_path, folder_list)
    if os.path.exists(save_path):
        print(save_path + " exists")
        #continue
    filecheck(save_path)
    for img_list in imgs_list:
        img = cv2.imread(os.path.join(folder_path,img_list))
        img2 = copy.deepcopy(img)
        print(os.path.join(folder_path,img_list))
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

        for ma in range(len(mask)):
            for m in range(len(mask[ma])):
                if mask[ma,m] == 0:
                    c_red[ma,m] = 0
                    c_green[ma,m] = 0
                    c_blue[ma,m] = 0                   


        # merge with mask got on one of a previous steps
        img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))
        #img3 = copy.deepcopy(img_a)


        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        numpy_vertical = np.hstack((img, img_a))
        
        img_a = (img_a*255).astype(np.uint8)
        img3 = copy.deepcopy(img_a)
        
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image',draw_circle)
        
        
        while(1):
            cv2.imshow('image', numpy_vertical)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'):
                mode = not mode
            elif k == ord('s'):
                try:
                    image_new = Image.fromarray((image_new).astype(np.uint8))
                    #cv2.imwrite(save_path+'/'+img_list.split('.')[0]+'.png', image_new)
                except:
                    pass
                b,g,rr,a = image_new.split()
                image_new = Image.merge("RGBA", (rr,g,b,a))
                image_new = remove_back(image_new)
                image_new.save(save_path+'/'+img_list.split('.')[0] + '.png', 'PNG')
                break
            elif k == 27:
                break
        
        cv2.destroyAllWindows()
        
        #cv2.imwrite(save_path+'/'+img_list.split('.')[0]+'.png', img_a)            

        # save to disk
        #cv2.imwrite(save_path+'/'+img_list.split('.')[0]+'.png', img_a*255)
        #cv2.imwrite('./3.png', img_a*255)


