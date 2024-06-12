import matplotlib.pyplot as plt
import cv2

# Load the image using OpenCV
image = cv2.imread('sabicard.jpeg')

# Convert the image from BGR to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.imshow(image_rgb)
plt.axis('off')  # Hide axis
plt.show()
