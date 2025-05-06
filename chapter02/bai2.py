import cv2
import numpy as np

img1 = np.ones((512,512,3),np.uint8)*0
cv2.circle(img1,(256,256),200,(255,255,255),-1)
img2 = np.ones((512,512,3),np.uint8)*0
pts1 = np.array([[256,10],[327,171],[195,171]],np.int32)
cv2.fillPoly(img2,[pts1],(255,255,255))

pts2 = np.array([[500,171],[327,171],[371,281]],np.int32)
cv2.fillPoly(img2,[pts2],(255,255,255))

pts3 = np.array([[427,437],[371,281],[256,374]],np.int32)
cv2.fillPoly(img2,[pts3],(255,255,255))

pts4 = np.array([[100,445],[141,281],[256,374]],np.int32)
cv2.fillPoly(img2,[pts4],(255,255,255))

pts5 = np.array([[12,171],[195,171],[141,281]],np.int32)
cv2.fillPoly(img2,[pts5],(255,255,255))

bitwise_xor = cv2.bitwise_xor(img1,img2)
white_background = np.ones((512, 512, 3), np.uint8) * 0
rows, cols, _ = white_background.shape
pixels = int(rows * cols * 0.02)
for _ in range(pixels):
    x = np.random.randint(0, cols)
    y = np.random.randint(0, rows)
    white_background[y, x] = [255,255,255]
final_image = cv2.bitwise_xor(bitwise_xor, white_background)
kernel12 = np.ones((3,3),np.uint8)
image1= cv2.morphologyEx(final_image, cv2.MORPH_OPEN, kernel12)
image2 = cv2.morphologyEx(final_image, cv2.MORPH_CLOSE, kernel12)
cv2.imshow("kq",final_image)
cv2.imshow("kq1",image1)
cv2.imshow("kq2",image2)
cv2.waitKey()
cv2.destroyAllWindows()