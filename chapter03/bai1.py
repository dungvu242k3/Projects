import cv2
import numpy as np

img1 = cv2.imread("C:/Users/dungv/Projects/Hand_On_Ml_Project_withOpenCV/chapter03/image-dog.jpg")
img2 = cv2.imread("C:/Users/dungv/Projects/Hand_On_Ml_Project_withOpenCV/chapter03/image-cat.jpg")

kernel1 = np.linspace(1, 0, img1.shape[1]).reshape(1, -1).repeat(img1.shape[0], axis=0)
kernel2 = np.linspace(0, 1, img2.shape[1]).reshape(1, -1).repeat(img2.shape[0], axis=0)

kernel1 = np.dstack([kernel1] * 3)
kernel2 = np.dstack([kernel2] * 3)

kq= (kernel1 * img1 + kernel2 * img2).astype(np.uint8)

cv2.imshow("kq", kq)
cv2.waitKey(0)
cv2.destroyAllWindows()