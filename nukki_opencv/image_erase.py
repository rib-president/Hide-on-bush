import cv2
import numpy as np
import copy

drawing = False # true if mouse is pressed
ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode, image_new

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),15,(0,200,0),-1)
            image_new = cv2.addWeighted(img, 1, img2, 0, 0)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        cv2.circle(img,(x,y),15,(0,200,0),-1)
        image_new = cv2.addWeighted(img, 1, img2, 0, 0)

    
            
img = cv2.imread('./exactly_cute.png',cv2.IMREAD_UNCHANGED)
img2 = copy.deepcopy(img)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('''define save_path'''+'.png', image_new)        

cv2.destroyAllWindows()
cv2.imwrite('./result.png', image_new)        
