import cv2
import numpy as np


def random_flip(image):
    code = np.random.choice([-1,0,1])
    flip_image =cv2.flip(image,code)
    return flip_image

def random_shift(image):
    rows, cols = image.shape[:2]
    shift_x = int(np.random.uniform(-0.2, 0.2) * cols)
    shift_y = int(np.random.uniform(-0.2, 0.2) * rows)
    shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, shift_matrix, (cols, rows))
    return shifted_image

def random_rotation(image):
    rows, cols = image.shape[:2]
    angle = np.random.uniform(-45,45)
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    return rotated_image

def random_brightness(image):
    brightness_factor = 1 + np.random.uniform(-0.5, 0.5)
    brightened_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return brightened_image

image = cv2.imread("C:/Users/dungv/Projects/Hand_On_Ml_Project_withOpenCV/chapter03/image-dog.jpg")

image1 = image.copy()
image1 = random_flip(image1)
cv2.imshow('flip',image1)
#cv2.waitKey(0)
image1 = random_shift(image1)
cv2.imshow('shift',image1)
#cv2.waitKey(0)
image1 = random_rotation(image1)
cv2.imshow('rotation',image1)
#cv2.waitKey(0)
image1 = random_brightness(image1)
cv2.imshow('brightness',image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
