import cv2
import matplotlib.pyplot as plt
# Load two images
img1 = cv2.imread('c/:anh3.png')
img2 = cv2.imread('C:/anh4.png')
# Convert the images to RGB color space
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)
# Display the images on the subplots
ax1.imshow(img1)
ax1.set_title('Image 1')
ax2.imshow(img2)
ax2.set_title('Image 2')
# Show the figure
plt.show()  