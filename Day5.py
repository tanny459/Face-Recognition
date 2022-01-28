import cv2
import face_recognition
from face_recognition.api import face_distance, face_encodings
import numpy as np


# Photo detection

'''image = face_recognition.load_image_file(
    "D:\\Python.vs\\Face_Recoginition\depp1.jpg")
image_encodings = face_recognition.face_encodings(image)[0]
image_locations = face_recognition.face_locations(image)[0]


image1 = face_recognition.load_image_file(
    "D:\\Python.vs\\Face_Recoginition\depp2.jpg")
image1_encodings = face_recognition.face_encodings(image1)[0]
image1_locations = face_recognition.face_locations(image1)[0]


results = face_recognition.compare_faces([image1_encodings], 
                                        image_encodings)
dist = face_recognition.face_distance([image1_encodings],
                                      image_encodings)


if results:


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    #print(results, image_locations)


    cv2.rectangle(image1,
                  (image1_locations[3], image1_locations[0]),
                  (image1_locations[1], image1_locations[2]),
                  (0, 255, 0),
                  2)


    cv2.putText(image1, f"{results}",
                (60, 60),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("image", image)
    cv2.imshow("image1", image1)

else:


    print(f"Can't detect anything the calculate distance was {dist} and result was {results}")


cv2.waitKey(0)

'''


# video detection


downy = face_recognition.load_image_file(
    "D:\\Python.vs\\Face_Recoginition\downy1.jpg")  # Path of the file
downy_encodings = face_recognition.face_encodings(downy)[0]
downy_locations = face_recognition.face_locations(downy)[0]


depp = face_recognition.load_image_file(
    "D:\\Python.vs\\Face_Recoginition\depp2.jpg") # Path of the file
depp_encodings = face_recognition.face_encodings(depp)[0]






known_face_encodings = [
    downy_encodings,
    depp_encodings
]


known_face_names = [
    "Robert Downey",
    "Johnny Depp "
]


cap = cv2.VideoCapture(0)


while True:
    flag, frame = cap.read()
    if not flag:
        print("Could not access the camera")
        break


    small_frame = cv2.resize(frame, (0, 0), fx=1/4, fy=1/4)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations)


    face_name = []
    for face_encodings in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encodings)
        name = "Unknown"
        face_distance = face_recognition.face_distance(
            known_face_encodings, face_encodings)
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_name.append(name)
    print(face_name)


    for (top, right, bottom, left), name in zip(face_locations, face_name):
        top    *= 4
        right  *= 4
        bottom *= 4
        left   *= 4
        cv2.rectangle(frame,
                      (left, top),
                      (right, bottom),
                      (0, 255, 0),
                      2) 
        cv2.rectangle(frame,
                      (left, bottom - 35),
                      (right, bottom),
                      (0, 255, 0),
                      cv2.FILLED) 
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1, (255, 255, 255), 1)
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
