import cv2
import matplotlib.pyplot as py
import numpy as np


# Playing with colors


image = cv2.imread("D:\\Python.vs\\Face_Recoginition\iron_man.jpg", 1)
print(image)
print(image.shape)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
py.figure(figsize = (16, 8))
py.subplot(1, 2, 1)
py.title("Grayed Image")
py.imshow(gray_image)


# py.subplot(1, 2, 2)
# py.title("Image")
# py.imshow(image)
# py.show()

# split channel

# image = cv2.imread("D:\\Python.vs\\Face_Recoginition\iron_man.jpg", 1)
# zeros = np.zeros((image.shape[0], image.shape[1]), np.uint8)
# b,g,r = cv2.split(image)
# print("Blue Channel: ", b)
# print("Green Channel: ", g)
# print("Red Channel: ", r)
# blue = cv2.merge([b, zeros, zeros])
# green = cv2.merge([zeros, g, zeros])
# red = cv2.merge([zeros, zeros, r])

# custom_image =  cv2.merge([b, g, r])
#custom_image =  cv2.merge([b, g+50, r])
# cv2.imshow("custom_image", custom_image)

# cv2.imshow("Blue", blue)
# cv2.imshow("Red", red)
# cv2.imshow("Green", green)

# one way

# image = cv2.imread("D:\\Python.vs\\Face_Recoginition\iron_man.jpg", 1)
# gray_image = cv2.imread("D:\\Python.vs\\Face_Recoginition\iron_man.jpg", 0)
# cv2.imshow("Frame Iron_man", image)
# cv2.imshow("Frame Iron_man Gray", gray_image)

# other way

# image = cv2.imread("D:\\Python.vs\\Face_Recoginition\iron_man.jpg")
# gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Frame Iron_man", image)
# cv2.imshow("Frame Iron_man Gray", gray_image)

# cv2.waitKey(0)

# Drawing



# Video implementation

# cap = cv2.VideoCapture("D:\\Python.vs\\Face_Recoginition\ideo.mp4")
cap = cv2.VideoCapture(0)
while cap.isOpened():
    sucess, frame = cap.read()
    print(sucess)
    if sucess:
        print("Yay!! We got tyhe video")
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Gray Frame', gray_frame)
        cv2.imshow('Frame', frame)
        k = cv2.waitKey(100)
        if k & 0xff == ord('q'):
            break
    else:
        print("sad lyf:(:(:(:(")
        break

cap.release()
cv2.destroyAllWindows()
