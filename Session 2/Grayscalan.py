import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize
scale = 80
width = int(img_gray.shape[1] * scale / 100)
height = int(img_gray.shape[0] * scale / 100)
dim = (width, height)
img_gray = cv2.resize(img_gray, dim, interpolation=cv2.INTER_AREA)

print(f"Original: {img.shape}")
print(f"Gray: {img_gray.shape}")

# Gray & Show
cv2.imshow('Gray Image', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.show()