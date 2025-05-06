import cv2
import numpy as np

img = cv2.imread("C:/Users/dungv/Projects/Hand_On_Ml_Project_withOpenCV/chapter03/image1.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernely = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

kernel_main = np.array([[0, 1, 2],
                       [-1, 0, 1],
                       [-2, -1, 0]])
kernel_secondary = np.array([[-2, -1, 0],
                            [-1, 0, 1],
                            [0, 1, 2]])

kernel = np.ones((2, 2)) / 6

sobel = cv2.filter2D(img,cv2.CV_64F,kernel)

sobelx = cv2.filter2D(sobel,cv2.CV_64F,kernelx)
sobely = cv2.filter2D(sobel,cv2.CV_64F,kernely)


sobel_main = cv2.filter2D(img,cv2.CV_64F,kernel_main)
sobel_secondary = cv2.filter2D(img,cv2.CV_64F,kernel_secondary)

main_sobel = np.sqrt(sobelx**2 + sobely**2)
secondary_sobel = np.sqrt(sobel_main**2 + sobel_secondary**2)
sobel_combined = np.sqrt(sobelx**2 + sobely**2 + sobel_main**2 + sobel_secondary**2)

main_sobel = cv2.convertScaleAbs(main_sobel)
secondary_sobel = cv2.convertScaleAbs(secondary_sobel)
sobel_combined = cv2.convertScaleAbs(sobel_combined)


#cv2.imshow("main_sobel",main_sobel)
cv2.imshow("secondary_sobel",secondary_sobel)
#cv2.imshow("sobel_combined",sobel_combined)
cv2.waitKey()
cv2.destroyAllWindows()