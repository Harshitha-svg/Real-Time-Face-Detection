import cv2  # Importing the OpenCV library for computer vision tasks
import streamlit as st  # Importing Streamlit for building interactive web applications
import numpy as np  # Importing NumPy for numerical computing
from PIL import Image  # Importing the Python Imaging Library for image processing



# Function to detect faces in live camera stream
def detect_faces():
    # Create the haar cascade for face detection using a pre-trained XML file
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # Explanation:
    # Pre-trained Model: The term "pre-trained model" refers to a machine learning model that has already been trained on a large dataset for a specific task. In this case, the pre-trained model is for face detection.
    # Haar Cascade Classifier: Haar Cascade is a machine learning-based approach that is used to identify objects in images or video streams. It works by training a cascade function on positive and negative images. The cascade function contains multiple stages, each of which contains a set of classifiers.
    # haarcascade_frontalface_default.xml: This XML file contains the trained data for the Haar cascade classifier specifically designed for frontal face detection. It consists of a set of features and weights that the classifier uses to identify whether a particular region of an image contains a face or not.
    # cv2.CascadeClassifier: This is a function provided by the OpenCV library that loads a cascade classifier from a file. It takes the path to the XML file as input and returns a cascade classifier object.
    # cv2.data.haarcascades: This is a predefined path in OpenCV where the pre-trained Haar cascade classifiers are stored. It contains XML files for various objects such as faces, eyes, and smiles.



    # Open the default camera (0) for capturing video
    cap = cv2.VideoCapture(0)

    # Infinite loop to continuously capture frames from the camera
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the captured frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,  # Parameter specifying how much the image size is reduced at each image scale
            minNeighbors=5,  # Parameter specifying how many neighbors each candidate rectangle should have to retain it
            minSize=(30, 30)  # Minimum possible object size. Objects smaller than this will be ignored
        )

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame with face detection
        cv2.imshow('Face Detection', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera capture
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()


# Function to detect faces in an uploaded image
def detect_faces_in_image(uploaded_image):
    # Convert the uploaded image file to a NumPy array
    img_array = np.array(Image.open(uploaded_image))

    # Create the haar cascade for face detection using a pre-trained XML file
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Detect faces in the grayscale image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,  # Parameter specifying how much the image size is reduced at each image scale
        minNeighbors=5,  # Parameter specifying how many neighbors each candidate rectangle should have to retain it
        minSize=(30, 30)  # Minimum possible object size. Objects smaller than this will be ignored
    )

    # Draw a rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting image with face detection
    st.image(img_array, channels="BGR", use_column_width=True)



#=================================Streamlit App========================================
# Streamlit UI
st.title("Face Detection")
st.subheader("Either Open Camera And Detect Faces Or Upload An Image And Detect Faces ")

# Button to start face detection in live camera stream
if st.button("Open Camera"):
    detect_faces()

# File uploader for detecting faces in an uploaded image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    detect_faces_in_image(uploaded_image)
