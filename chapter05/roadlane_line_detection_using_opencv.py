import cv2
import numpy as np


def roi_mask(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255)
    masked_img = cv2.bitwise_and(img,mask)
    return masked_img
def hough_lines(img,rho,theta,threshold,min_line_len,max_line_gap):
    lines = cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),minLineLength = min_line_len,maxLineGap = max_line_gap)
    return lines
def draw_lines(img,lines,color = [255,0,0],thickness = 5) :
    for line in lines :
        for x1,y1,x2,y2 in line :
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)
cap = cv2.VideoCapture("C:/road.mp4")
while True :
    ret,frame = cap.read()
    if ret :
        try :
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(5,5),0)
            edges = cv2.Canny(blur,50,150)
            roi_vertices = np.array([[(0,frame.shape[0]),(frame.shape[1]//2),(frame.shape[0]//2 + 50),(frame.shape[1],frame.shape[0])]],dtype = np.int32)
            roi = roi_mask(edges,roi_vertices)
            lines = hough_lines(roi,rho = 2,theta = np.pi/180,threshold = 50,min_line_len = 100, max_line_gap = 50)
            line_img = np.zeros((frame.shape[0],frame[1],3),dtype = np.uint8)
            draw_lines(line_img,lines)
            result = cv2.addWeighted(frame,0.8,line_img,1,0)
            cv2.imshow("kq",result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except: pass
    else :
        break
cap.release()
cv2.destroyAllWindows()