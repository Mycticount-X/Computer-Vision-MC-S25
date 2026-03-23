import cv2
import matplotlib.pyplot as plt

def edge_all(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize
    scale = 50
    wid = int(img.shape[1] * scale / 100)
    hei = int(img.shape[0] * scale / 100)

    size = (wid, hei)
    img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
    
    # Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Sobel
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, dx=0, dy=1, ksize=3)

    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    sobel_duo = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    # or sobel_duo = cv2.bitwise_or(sobelx, sobely)

    # Canny
    canny = cv2.Canny(blurred, 100, 200)

    # Laplacian
    laplace = cv2.Laplacian(blurred, cv2.CV_64F)
    laplace = cv2.convertScaleAbs(laplace)

    result = [img, sobelx, sobely, sobel_duo, canny, laplace]
    result_desc = ['Grayscale', 'Sobel X', 'Sobel Y', 'Sobel Duo', 'Canny', 'Laplacian']

    plt.figure(figsize=(6, 5))
    for i, (res_img, res_desc) in enumerate(zip(result, result_desc)):
        plt.subplot(2, 3, i+1)
        plt.imshow(res_img, cmap='gray')
        plt.title(res_desc)
        plt.xticks([])
        plt.yticks([])
    
    plt.show()


if __name__ == '__main__':
    image = cv2.imread('./flower.jpg')
    edge_all(image)

