import cv2
import numpy as np

img = cv2.imread("C:/Users/dungv/Projects/Hand_On_Ml_Project_withOpenCV/chapter04/geometric_shapes.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY_INV)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for i in range (68) :
    if 0 <= i <= 6 :
        cv2.drawContours(img, contours, i, (0, 100, 0), 3)
    elif i == 29 :
        cv2.drawContours(img, contours, i, (0, 100, 0), 5) 
    elif 7 <= i <= 11 or i == 24 :
        cv2.drawContours(img, contours, i , (255,0,0), 3) 
    elif i == 30 :
        cv2.drawContours(img, contours, i , (255,0,0), 5) 
    elif 12 <= i <= 17 or 25 <= i <= 26 :
        cv2.drawContours(img, contours, i , (255,0,255), 3) 
    elif i == 28 :
        cv2.drawContours(img, contours, i , (255,0,255), 5) 
    elif 18 <= i <= 23 or i == 27 :
        cv2.drawContours(img, contours, i, (153,136,119), 3) 
    elif i == 31 :
        cv2.drawContours(img, contours, i, (153,136,119), 5) 
    elif 45 <= i <=51 or i == 62 :
        cv2.drawContours(img, contours, i, (105,105,105),3) 
    elif i == 33 :
        cv2.drawContours(img, contours, i, (105,105,105),5)
    elif 36 <= i <= 43 :
        cv2.drawContours(img, contours, i, (0,140,255),3) 
    elif i == 34 :
        cv2.drawContours(img, contours, i, (0,140,255),5) 
    elif i == 44 or 52 <= i <=56 or 63 <= i <= 65:
        cv2.drawContours(img, contours, i, (144,96,205),3) 
    elif i == 32 :
        cv2.drawContours(img, contours, i, (144,96,205),5) 
    elif 57 <= i <=61 or 66 <=i<=67 :
        cv2.drawContours(img, contours, i, (255,255,100),3)
    elif i == 35 :
        cv2.drawContours(img, contours, i, (255,255,100),5)


#cv2.imshow("edges",thresh) 
cv2.imshow("kq",img)
cv2.waitKey()
cv2.destroyAllWindows()
        