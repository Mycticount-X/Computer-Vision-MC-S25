import cv2
import matplotlib.pyplot as plt
import numpy as np

def filter_all(img, blur=11):
    kernel = np.ones((blur, blur), np.float32) / (blur * blur)

    avging = cv2.blur(img, (blur, blur))
    blur_filter = cv2.filter2D(img, -1, kernel)
    median = cv2.medianBlur(img, blur)
    gaussian = cv2.GaussianBlur(img, (blur, blur), 0)
    bilateral = cv2.bilateralFilter(img, blur, 75, 75)

    result = [img, avging, blur_filter, median, gaussian, bilateral]
    result_desc = ['original', 'averaging', 'blur', 'median', 'gaussian', 'bilateral']

    plt.figure(figsize=(6, 5))
    for i, (res_img, res_desc) in enumerate(zip(result, result_desc)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(res_img)
        plt.title(res_desc)
        plt.xticks([])
        plt.yticks([])

    plt.show()

if __name__ == '__main__':
    image = cv2.imread('./flowers.jpg')
    filter_all(image)