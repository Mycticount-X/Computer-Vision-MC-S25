import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Original: {img.shape}")
print(f"Gray: {img_gray.shape}")

cv2.imshow('Gray Image', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.show()