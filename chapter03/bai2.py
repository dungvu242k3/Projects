import cv2
import numpy as np

img = cv2.imread("C:/Users/dungv/Projects/Hand_On_Ml_Project_withOpenCV/chapter03/image1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

box_kernel = np.ones((1,1))/2
kernel = np.ones((3,3))/9

kernelx = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])
kernely = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

sobel = cv2.filter2D(img,cv2.CV_64F,kernel)
box_sobel = cv2.filter2D(img,cv2.CV_64F,box_kernel)
sobelx = cv2.filter2D(img, cv2.CV_64F, kernelx)
sobely = cv2.filter2D(img, cv2.CV_64F, kernely)

main_sobel = np.sqrt(sobelx**2 + sobely**2)
main_sobel = cv2.convertScaleAbs(main_sobel)

sobel_combied = np.sqrt(box_sobel**2 + sobelx**2)
sobel_combied = cv2.convertScaleAbs(sobel_combied)

secondary_sobel = np.sqrt(sobel**2 + sobely**2)
secondary_sobel = cv2.convertScaleAbs(secondary_sobel)

cv2.imshow("main_sobel",main_sobel)
cv2.imshow("secondary_sobel",sobel_combied)
#cv2.imshow("sobel_combined",secondary_sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()
