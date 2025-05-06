import cv2
import numpy as np

image = cv2.imread("C:/Users/dungv/Projects/Hand_On_Ml_Project_withOpenCV/chapter04/geometric_shapes.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def get_contour_color(img, contour):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=3)

    mean_color = cv2.mean(img, mask=mask)
    
    return mean_color

for contour in contours:
    color = get_contour_color(image, contour)
    color = tuple(map(int, color))
    cv2.drawContours(image, [contour], -1, color, thickness=3)

cv2.imshow("kq", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
