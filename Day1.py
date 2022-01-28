import cv2 
import matplotlib.pyplot as plt

image = cv2.imread("D:\\Python.vs\\Face_Recoginition\iron_man.jpg", 1) # for grey use 0 insted of 1
print(image)
cv2.imshow("Iron_man Frame",image)
image[1:4:2]
cv2.waitKey(0)
# print(image)
# print(image.Shape)
#  plt.imshow(cv2.cvtColor(image.cv2.COLOR_BGR2EBG))


cap = cv2.VideoCapture(0)

while True:
    flag, frame = cap.read()
    if not flag:
        print("Could not access the camera")
        break
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()