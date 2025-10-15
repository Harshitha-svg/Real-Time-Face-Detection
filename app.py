import cv2  # OpenCV for computer vision
import streamlit as st  # Streamlit for web interface
import numpy as np  # NumPy for numerical operations
from PIL import Image  # For handling image files

# -------------------- Function 1: Detect Faces in Uploaded Image --------------------
def detect_faces_in_image(uploaded_image):
    img = Image.open(uploaded_image)
    img_array = np.array(img)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert RGB → Grayscale for detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(img_array, channels="RGB", use_column_width=True)
    st.success(f"✅ Detected {len(faces)} face(s).")


# -------------------- Function 2: Detect Faces from Streamlit Camera --------------------
def detect_faces_from_camera_input(picture):
    img = Image.open(picture)
    img_array = np.array(img)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(img_array, channels="RGB", use_column_width=True)
    st.success(f"✅ Detected {len(faces)} face(s) from your camera input.")


# -------------------- Streamlit UI --------------------
st.title("📸 Face Detection App")
st.subheader("Choose an option below to detect faces:")

# Option 1: Live Camera (only works when running locally, not on Streamlit Cloud)
if st.button("Open Live Camera"):
    if not st.runtime.is_streamlit_cloud():
        detect_faces()
    else:
        st.warning("🚫 Live webcam access isn't supported on Streamlit Cloud. Try uploading an image or using the camera input below.")

# Option 2: Upload an image
uploaded_image = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    detect_faces_in_image(uploaded_image)

# Option 3: Use Streamlit's built-in camera input
st.markdown("---")
st.subheader("🎥 Or take a picture using your webcam:")
picture = st.camera_input("Take a picture")
if picture:
    detect_faces_from_camera_input(picture)


st.markdown("---")
st.info("Developed by Harshitha 💻 | Powered by OpenCV + Streamlit")


