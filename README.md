

# 👁️ Real-Time Face Detection using OpenCV & Streamlit

This project demonstrates **real-time face detection** using **OpenCV** and **Streamlit**. It captures live video from your webcam and detects faces in real-time using **Haar Cascade Classifier**.

---

## 🚀 Features

* Real-time webcam video streaming
* Automatic face detection using Haar cascade
* User-friendly Streamlit interface
* Option to capture and display detected faces
* Lightweight and runs directly in the browser

---

## 🧠 Technologies Used

| Library          | Purpose                              |
| ---------------- | ------------------------------------ |
| **OpenCV**       | For video capture and face detection |
| **Streamlit**    | For building the interactive web app |
| **NumPy**        | For image array manipulation         |
| **PIL (Pillow)** | For image processing                 |

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<Harshitha-svg>/real-time-face-detection.git
cd real-time-face-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

* **Windows:** `venv\Scripts\activate`
* **Mac/Linux:** `source venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501/`).

---

## 📁 Project Structure

```
📦 real-time-face-detection
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # List of dependencies
├── haarcascade_frontalface_default.xml  # Pre-trained model for face detection
├── README.md               # Project documentation
└── images/                 # (Optional) Folder to store captured images
```


## 🧩 How It Works

1. The app uses **OpenCV** to access your webcam feed.
2. It loads the **Haar Cascade Classifier** for frontal face detection.
3. Each frame is analyzed — if a face is detected, a bounding box is drawn around it.
4. Streamlit displays the processed frames in real-time through a web interface.

---


## ⚙️ requirements.txt

If you don’t have one, include the following:

```
streamlit
opencv-python
numpy
Pillow
```

---

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify it for educational purposes.

---
