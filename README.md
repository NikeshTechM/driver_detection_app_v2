
````markdown
# Face Recognition Flask App

This Flask application streams live face recognition and face mesh detection video feed from multiple sources: webcam, single image, folder of images, or video file. It uses MediaPipe for face mesh and face detection, and a trained KNN classifier for face recognition.

---

## Features

- Supports input sources:
  - Webcam (`camera`)
  - Single image (`image:/path/to/image.jpg`)
  - Folder of images (`folder:/path/to/folder`)
  - Video file (`video:/path/to/video.mp4`)
- Streams processed frames in real-time (or simulated real-time for images/folders) to a web page.
- Displays face mesh landmarks.
- Recognizes known faces using a pre-trained KNN model.
- Socket server to query recognized face name remotely.

---

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- MediaPipe
- face_recognition
- Flask
- numpy (usually a dependency of above)

Install dependencies with:

```bash
pip install opencv-python mediapipe face_recognition flask numpy
````

---

## Setup

1. Place your trained KNN model file `trained_knn_model.clf` in the working directory.

2. Prepare your input source (camera, image, folder, or video).

---

## Usage

Run the app with the following options:

```bash
python app.py --source [source] --port [port] --timer [timer_value]
```

### Source options and example commands:

* **Webcam (default)**

  ```bash
  python app.py --source camera
  ```

* **Single image (loops the image to simulate video feed)**

  ```bash
  python app.py --source image:/full/path/to/image.jpg
  ```

* **Folder of images (loops through images repeatedly)**

  ```bash
  python app.py --source folder:/full/path/to/image_folder
  ```

* **Video file**

  ```bash
  python app.py --source video:/full/path/to/video.mp4
  ```

### Additional options:

* `--port`: Port for Flask app (default 5000)

  Example:

  ```bash
  python app.py --source camera --port 8080
  ```

* `--timer`: Initial timer value (default 10000, usage depends on your app logic)

  Example:

  ```bash
  python app.py --source camera --timer 15000
  ```

---

## How it works

* The app reads frames from the selected source.
* Runs face detection and face mesh with MediaPipe.
* Performs face recognition using your trained KNN model.
* Overlays results on frames.
* Streams frames over HTTP to the browser as MJPEG video.
* Single images and image folders are looped continuously to simulate live video.

---

## Accessing the app

Open your browser and navigate to:

```
http://localhost:5000/
```

(Or replace `5000` with your chosen port if different.)

You will see the live video feed with face landmarks and recognition results.

---

## Socket server

A socket server runs on port `42000` to allow external clients to query the recognized user name by sending a message containing the keyword `spit`.

---

## Troubleshooting

* Make sure your camera is accessible if using webcam.
* Ensure image/video paths are correct.
* Folder must contain valid image files (`.jpg`, `.jpeg`, `.png`).
* If you get errors loading your trained model, verify the `trained_knn_model.clf` file exists and is compatible.



## Author

Ayush