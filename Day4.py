# Face Detection


import numpy as np
import cv2
import mediapipe as mp

# Drawing utility

mp_drawing = mp.solutions.drawing_utils

# Face Detection utility

mp_face_detection = mp.solutions.face_detection

# Model Detection utility

model_detection = mp_face_detection.FaceDetection()

# Model Face_Mesh

mp_face_mesh = mp.solutions.face_mesh
model_face_mesh = mp_face_mesh.FaceMesh()
drawing_spec = mp_drawing.DrawingSpec((255, 0, 0), thickness = 1, circle_radius = 1)



'''cap = cv2.VideoCapture(0)

while cap.isOpened():
    flag, frame = cap.read()
    if not flag:
        print("Could not access the camera")
        break
        
# Face Detection

    result = model_detection.process(frame)
    for landmark in result.detections:
        print(mp_face_detection.get_key_point(
            landmark, mp_face_detection.FaceKeyPoint.NOSE_TIP)) 
        mp_drawing.draw_detection(frame, landmark)
    print(result.detections)

# Fcae Mesh

    result = model_face_mesh.process(frame)
    for landmark in result.multi_face_landmarks:
        print(landmark)
        mp_drawing.draw_landmarks(
            image = frame,
            landmark_list = landmark,
            connections = mp_face_mesh.FACE_CONNECTIONS,
            landmark_drawing_spec = drawing_spec,
            connection_drawing_spec = drawing_spec)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
model = mp_selfie_segmentation.SelfieSegmentation(model_selection = 1)
bg_image = cv2.imread("Downloads:\\mahadev.jpg", 1)


cap = cv2.VideoCapture(0)

while True:
    flag, frame = cap.read()
    if not flag:
        print("Could not access the camera")
        
    results = model.process(frame)
    cond = np.stack((results.segmentation_mask, ) * 3, axis = -1) > 0.1
    if bg_image is None:
        bg_image = np.zeros(frame.shape, dtype = np.uint8)
        bg_image[:] = (0, 0, 225)
    bg_image = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
    output_image = np.where(cond, frame, bg_image)

    cv2.imshow('Frame', results.segmentation_mask)
    cv2.imshow('Frame', output_image)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()