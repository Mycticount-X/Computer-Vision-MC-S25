import cv2
import matplotlib.pyplot as plt
import numpy as np

def threshold_all(img, tres=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, tres, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(gray, tres, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(gray, tres, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(gray, tres, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(gray, tres, 255, cv2.THRESH_TOZERO_INV)
    ret, thresh6 = cv2.threshold(gray, tres, 255, cv2.THRESH_OTSU)
    ret, thresh7 = cv2.threshold(gray, tres, 255, cv2.THRESH_TRIANGLE)

    result = [
        gray, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6, thresh7
    ]

    result_desc = [
        'gray', 'BINARY', 'BINARY_INV', 'TRUNC',
        'TOZERO', 'TOZERO_INV', 'OTSU', 'TRIANGLE'
    ]

    plt.figure(figsize=(6, 5))
    for i, (res_img, res_desc) in enumerate(zip(result, result_desc)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(res_img, cmap='gray')
        plt.title(res_desc)
        plt.xticks([])
        plt.yticks([])

    plt.show()

if __name__ == '__main__':
    image = cv2.imread('./flowers.jpg')
    threshold_all(image)