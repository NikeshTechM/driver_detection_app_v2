import cv2
import mediapipe as mp
import time
import argparse
import pickle
import face_recognition
import os
from flask import Flask, render_template, Response, jsonify
import warnings
import socket
import threading
from collections import deque
import logging
import traceback

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to INFO or WARNING to reduce verbosity if needed
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---- Argument Parsing ----
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', default=5000, required=False)
parser.add_argument('-t', '--timer', default=10000, required=False)
parser.add_argument('-s', '--source', default='camera', required=False,
                    help='Options: "camera", "video:/path.mp4", "image:/path.jpg", "folder:/folder_path"')
args = vars(parser.parse_args())

port = int(args['port'])
initial_timer_value = int(args['timer'])
source = args['source']

source_type = 'camera'
source_path = None

if source.startswith('image:'):
    source_type = 'image'
    source_path = source.split('image:')[1]
elif source.startswith('video:'):
    source_type = 'video'
    source_path = source.split('video:')[1]
elif source.startswith('folder:'):
    source_type = 'folder'
    source_path = source.split('folder:')[1]

# ---- Flask App Initialization ----
app = Flask(__name__)

# ---- Globals ----
font_scale = 0.5
position1 = (50, 50)
font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)
thickness = 1
linetype = cv2.LINE_AA
start_time = time.time()
fps = 0
count = 0
user = 'Driver'
name_list = deque(['Driver'] * 10, maxlen=10)
cam_width = 0
cam_height = 0

# ---- MediaPipe Setup ----
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

def get_face_mesh():
    return mp_face_mesh.FaceMesh(
        static_image_mode=(source_type != 'camera'),
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5 if source_type == 'camera' else 0.0
    )

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.85)

try:
    with open(os.path.join(os.getcwd(), "trained_knn_model.clf"), 'rb') as file:
        knn_clf = pickle.load(file)
    logging.info("KNN model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load KNN model: {e}")
    logging.error(traceback.format_exc())
    exit(1)

host = '0.0.0.0'
logging.info(f'Host IP: {host}')

# ---- Shared Camera Object and Lock ----
camera = None
camera_lock = threading.Lock()

# ---- Folder Image Reader Class ----
class FolderLoop:
    def __init__(self, folder_path):
        self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not self.files:
            raise ValueError("No image files found in folder.")
        self.index = 0
        self.repeat_frame_count = 0
        self.max_repeat = 15
        logging.info(f"FolderLoop initialized with {len(self.files)} images.")

    def read(self):
        img = cv2.imread(self.files[self.index])
        self.repeat_frame_count += 1
        if self.repeat_frame_count >= self.max_repeat:
            self.index = (self.index + 1) % len(self.files)
            self.repeat_frame_count = 0
        return True, img.copy()

# ---- Frame Transformation and Recognition ----
def transform(image):
    global count, start_time, fps, user

    count += 1
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if count % 5 == 0:
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = int(5 / elapsed_time)
        start_time = time.time()
    cv2.putText(image, f'FPS:{fps:.2f}', position1, font, font_scale, color, thickness, linetype)

    if count % 15 == 0:
        count = 1
        results = face_detection.process(image_rgb)
        if results.detections and len(results.detections) == 1:
            logging.debug("One face detected by MediaPipe Face Detection.")
            try:
                x = int(results.detections[0].location_data.relative_bounding_box.xmin * cam_width)
                w = int(results.detections[0].location_data.relative_bounding_box.width * cam_width)
                y = int(results.detections[0].location_data.relative_bounding_box.ymin * cam_height)
                h = int(results.detections[0].location_data.relative_bounding_box.height * cam_height)

                X_face_locations = [(y, x+w, y+h, x)]
                faces_encodings = face_recognition.face_encodings(image_rgb, known_face_locations=X_face_locations)

                if faces_encodings:
                    logging.debug(f"Found {len(faces_encodings)} face encoding(s).")
                    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
                    are_matches = [closest_distances[0][i][0] <= 0.45 for i in range(len(X_face_locations))]
                    predictions = [(pred, loc) if rec else ("unknown", loc)
                                   for pred, loc, rec in zip(knn_clf.predict(faces_encodings),
                                                             X_face_locations, are_matches)]

                    for name, _ in predictions:
                        name_list.append('Guest' if name == 'unknown' else name)
                        if len(set(list(name_list)[:3])) == 1:
                            user = name_list[1]
                            logging.info(f"Recognized user: {user}")
            except Exception as e:
                logging.error(f"Error during face recognition: {e}")
                logging.error(traceback.format_exc())

    # Use face mesh within a safe context
    try:
        with get_face_mesh() as face_mesh:
            results = face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                logging.debug(f"Face mesh landmarks detected: {len(results.multi_face_landmarks)} faces.")
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    except Exception as e:
        logging.error(f"Error during face mesh processing: {e}")
        logging.error(traceback.format_exc())

    return image

# ---- Recognition Thread ----
def recognition_loop():
    global camera, camera_lock
    while True:
        try:
            with camera_lock:
                success, frame = camera.read()
            if not success:
                logging.warning("Failed to read frame from camera.")
                continue
            frame_copy = frame.copy()
            transform(frame_copy)
            time.sleep(0.1)
        except Exception as e:
            logging.error(f"Error in recognition loop: {e}")
            logging.error(traceback.format_exc())

# ---- Video Streaming ----
def generate_frames():
    global camera, camera_lock
    while True:
        try:
            with camera_lock:
                success, frame = camera.read()
            if not success:
                logging.warning("Failed to read frame from camera.")
                continue
            frame_copy = frame.copy()
            output_frame = transform(frame_copy)
            ret, buffer = cv2.imencode('.png', output_frame)
            if not ret:
                logging.warning("Failed to encode frame.")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logging.error(f"Error in generate_frames: {e}")
            logging.error(traceback.format_exc())

# ---- Socket Server ----
def socket_server():
    try:
        server_socket = socket.socket()
        server_socket.bind((host, 42000))
        server_socket.listen(1)
        logging.info("[INFO] Socket server listening on port 42000")

        while True:
            conn, address = server_socket.accept()
            logging.info(f"Connection started from: {address}")
            try:
                while True:
                    data = conn.recv(1024).decode()
                    if not data:
                        break
                    if 'spit' in data.lower():
                        logging.info(f"From connected user: {data}")
                        conn.send(f'{user.capitalize()}\n'.encode())
            except Exception as e:
                logging.error(f"Error handling socket connection from {address}: {e}")
                logging.error(traceback.format_exc())
            finally:
                conn.close()
                logging.info(f"Connection closed from: {address}")
    except Exception as e:
        logging.error(f'[ERROR] Socket Server: {e}')
        logging.error(traceback.format_exc())
        exit(1)

# ---- Flask Routes ----
@app.route('/')
def index():
    return render_template('index.html', timer_value=initial_timer_value)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recognized_face', methods=['GET'])
def get_recognized_face():
    final_str = 'Unidentified User Detected' if user == 'Guest' else f'Welcome {user.capitalize()}'
    return jsonify({'faceName': final_str})

# ---- Main Entrypoint ----
if __name__ == "__main__":
    logging.info("[INFO] Starting system...")

    try:
        if source_type == 'camera':
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise RuntimeError("Could not open webcam")
            cam_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            cam_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"Camera initialized: width={cam_width}, height={cam_height}")

        elif source_type == 'video':
            if not os.path.isfile(source_path):
                raise FileNotFoundError(f"Video file not found: {source_path}")
            camera = cv2.VideoCapture(source_path)
            cam_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            cam_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"Video file opened: {source_path}")

        elif source_type == 'image':
            if not os.path.isfile(source_path):
                raise FileNotFoundError(f"Image file not found: {source_path}")
            image = cv2.imread(source_path)
            if image is None:
                raise ValueError(f"Failed to load image: {source_path}")
            cam_width = image.shape[1]
            cam_height = image.shape[0]

            class ImageLoop:
                def __init__(self, image):
                    self.image = image
                    self.repeat_frame_count = 0
                    self.max_repeat = 15
                    logging.info(f"ImageLoop initialized for {source_path}")

                def read(self):
                    self.repeat_frame_count += 1
                    if self.repeat_frame_count >= self.max_repeat:
                        self.repeat_frame_count = 0
                    return True, self.image.copy()

            camera = ImageLoop(image)

        elif source_type == 'folder':
            if not os.path.isdir(source_path):
                raise NotADirectoryError(f"Folder not found: {source_path}")
            camera = FolderLoop(source_path)
            sample = cv2.imread(camera.files[0])
            cam_width = sample.shape[1]
            cam_height = sample.shape[0]

        else:
            raise ValueError(f"Invalid source type: {source_type}")

    except Exception as e:
        logging.error(f"Error initializing video source: {e}")
        logging.error(traceback.format_exc())
        exit(1)

    threading.Thread(target=recognition_loop, daemon=True).start()
    threading.Thread(target=socket_server, daemon=True).start()
    threading.Thread(target=lambda: app.run('0.0.0.0', port, debug=False, use_reloader=False), daemon=True).start()

    while True:
        logging.info(f"Recognized: {user}")
        time.sleep(2)
