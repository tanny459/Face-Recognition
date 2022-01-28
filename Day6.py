import cv2
import numpy as np
import streamlit as st
from PIL import Image
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
model_face_mesh = mp_face_mesh.FaceMesh()


st.title("Open cv Operations")
st.subheader("Image operations")


add_selectbox = st.sidebar.selectbox(
    "What operations you would like to perform?",
    ("About", "Gray_Scale", "Blue_Image", "Green_Image", "Red_Image", "Meshing")
)


if add_selectbox == "About":
    st.write("This application is a demo for streamlit.")

elif add_selectbox == "Gray_Scale":
    image_file_path = st.sidebar.file_uploader("Upload Image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st.image(gray_image)


elif add_selectbox == "Blue_Image":
    image_file_path = st.sidebar.file_uploader("Upload Image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        blue_image = cv2.merge([zeros, zeros, b])
        st.image(blue_image)


elif add_selectbox == "Green_Image":
    image_file_path = st.sidebar.file_uploader("Upload Image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        green_image = cv2.merge([zeros, g, zeros])
        st.image(green_image)


elif add_selectbox == "Red_Image":
    image_file_path = st.sidebar.file_uploader("Upload Image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        r, g, b = cv2.split(image)
        red_image = cv2.merge([r, zeros, zeros])
        st.image(red_image)


elif add_selectbox == "Meshing":
    image_file_path = st.sidebar.file_uploader("Upload Image")
    if image_file_path is not None:
        image = np.array(Image.open(image_file_path))
        st.sidebar.image(image)
        results = model_face_mesh.process(image)

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image = image,
                landmark_list = face_landmarks,
                )
    
        st.image(image)
