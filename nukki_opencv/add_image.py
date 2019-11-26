import cv2
import numpy as np
import copy
import time
from PIL import Image
#ix,iy = -1,-1

r=63
drawing = False
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    global drawing
    #drawing = False
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        print(drawing)
        if drawing == True:            
            copy_bg_to_fg(x,y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        copy_bg_to_fg(x,y)


def copy_bg_to_fg(x,y):
    cv2.circle(img,(x,y),r,(255,0,0),-1)
    mask_img = img2[y-r:y+r, x-r:x+r]
    rgba = cv2.cvtColor(mask_img, cv2.COLOR_RGB2RGBA)
    mask = np.full((mask_img.shape[0], mask_img.shape[1]), 0, dtype=np.uint8)
    cv2.circle(mask,(r,r), r, (255,255,255),-1)
    fg = cv2.bitwise_or(rgba, rgba, mask=mask)                  
    add_img = Image.fromarray(fg)
    nukki_img.paste(add_img, (x-r, y-r), add_img)

       
# Create a black image, a window and bind the function to window
img = cv2.imread('./background.jpg', -1)
img2 = copy.deepcopy(img)
nukki_img = Image.open('./foreground_transparent.png')
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('s'):
        nukki_img.save('./asdkfjasdkfjaskdf.png', 'PNG')
        break
cv2.destroyAllWindows()


