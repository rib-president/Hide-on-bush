import numpy as np
import cv2

img = cv2.imread('background.jpg')
overlay_t = cv2.imread('foreground_transparent.png',-1)
x = 0
y = 0

background_img = img
img_to_overlay_t = overlay_t

bg_img = background_img.copy()


b,g,r,a = cv2.split(img_to_overlay_t)
overlay_color = cv2.merge((b,g,r))

mask = cv2.medianBlur(a,5)

h, w, _ = overlay_color.shape
roi = bg_img[y:y+h, x:x+w]

img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))

img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

