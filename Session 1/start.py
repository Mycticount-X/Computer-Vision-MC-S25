import cv2

img = cv2.imread('./flowers.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (900, 600))

img[0:300, :, 0] = 0
img[300:600, :, 1] = 0
img[600:900, :, 2] = 0

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img.shape)


