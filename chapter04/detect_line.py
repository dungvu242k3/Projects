import cv2
import numpy as np

img = cv2.imread("C:/Users/dungv/Projects/Hand_On_Ml_Project_withOpenCV/chapter04/chess_board.jpg")
img = cv2.resize(img,(640,480))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
lines = cv2.HoughLinesP(edges,rho = 1,theta = np.pi/180,threshold = 100,minLineLength = 50,maxLineGap = 10)
for line in lines :
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)
cv2.imshow("kq",img)
cv2.waitKey()
cv2.destroyAllWindows()