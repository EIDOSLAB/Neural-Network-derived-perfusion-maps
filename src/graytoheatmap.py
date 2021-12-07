import numpy as np
import cv2

file_name = 'samples/_90.png'
save_name = '_90_colored.png'

img = cv2.imread(file_name, 1)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
cv2.imwrite(save_name,heatmap_img)
