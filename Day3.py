import cv2
import matplotlib.pyplot as py
import numpy as np


#shifting matrix
 
image = cv2.imread("D:\\Python.vs\\Face_Recoginition\iron_man.jpg", 1)
m = np.float32([[1, 0, 50], [0, 1, 50]])
height, width, channel = image.shape
t_image = cv2.warpAffine(image, m , (height, width))
cv2.imshow("Image", image)
cv2.imshow("Translated Image", t_image)
cv2.waitKey(0)
 
#Rotate matrix
 
image = cv2.imread("D:\\Python.vs\\Face_Recoginition\iron_man.jpg", 1)
height, width, channel = image.shape
r_matrix = cv2.getRotationMatrix2D((height/4, width/2), 90, 1)
r_image = cv2.warpAffine(image, r_matrix, (height, width))
scaled_image = cv2.resize(image, None, fx = 4, fy = 2)
cv2.imshow("Image", image)
cv2.imshow("Roteted Image", r_image)
cv2.imshow("Scaled Image", scaled_image)
cv2.waitKey(0)

#Adding a background

image = cv2.imread("D:\\Python.vs\\Face_Recoginition\iron_man.jpg", 1)
image1 = cv2.imread("D:\\Python.vs\\Face_Recoginition\ckground.jpg", 1)
image = cv2.resize(image, (image1.shape[1], image1.shape[0]))
adding = image + image1
blend_image = cv2.addWeighted(image, 0.9, image1, 0.7, gamma = 0.1)
cv2.imshow("Added Image", adding)
cv2.imshow("Blended Image", blend_image)
cv2.waitKey(0)

#Convulation of the image

image = cv2.imread("D:\\Python.vs\\Face_Recoginition\ironman.jpg", 1)
kernel = np.float32([[9, 2, 0],
                      [0, 1, 1],
                      [4, 1, -5]])
conv_image = cv2.filter2D(image, -1, kernel)
cv2.imshow("conv", conv_image)
cv2.waitKey(0)

#now video

cap = cv2.VideoCapture(0)
image = cv2.imread("D:\\Python.vs\\Face_Recoginition\iron_man.jpg", 1)
while True:
    flag, frame = cap.read()
    if not flag:
        print("Could not access the webcam")
        break
    image = cv2.resize(image, (frame.shape[1], frame.shape[0]))
    blended_frame = cv2.addWeighted(frame, 0.6, image, 0.6, gamma = 0.4)
    cv2.imshow("Blended Frame", blended_frame)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()