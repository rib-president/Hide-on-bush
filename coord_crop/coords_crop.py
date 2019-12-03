# 2018.01.17 20:39:17 CST
# 2018.01.17 20:50:35 CST
import numpy as np
import cv2
import os


path = './image/'
img_list = sorted(os.listdir(path))
save = './'



i = 0

for img in img_list:
    pts = np.array([[20, 100],[50,50],[70,50],[150,100],[300,150],[500,100],[500,500],[300,500],[150,400],[70,400],[50,300],[20,200]])
    img_path = os.path.join(path, img)
    img = cv2.imread(img_path)

    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    #bg = np.ones_like(croped, np.uint8)*255
    #cv2.bitwise_not(bg,bg, mask=mask)
    #dst2 = bg+ dst


    #cv2.imwrite("croped.png", croped)
    cv2.imwrite("mask_%d.png" % i, mask)
    cv2.imwrite("{}/{:06d}.jpg".format(save, i), dst)
    #cv2.imwrite("dst2.png", dst2)
    i+=1
