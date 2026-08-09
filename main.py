"""
main.py
--------
ClassSentinel core pipeline: LBPH face recognition (attendance), YOLOv8
phone detection, MediaPipe EAR drowsiness detection.

Changes from the original version, based on real issues found in this
project's own attendance.csv:
  - All tunable values now live in config.py (no more different thresholds
    hardcoded in different files)
  - Student IDs/names are normalized (uppercase ID, title-case name) so
    "4bd23IS050" and "4BD23IS050" are treated as one person
  - Attendance writes to SQLite (single source of truth) via database.py,
    and is exported to attendance.csv at the end — so the CSV can never
    drift from the DB the way attendance.csv/attendence.csv did before
  - Won't re-mark someone already marked earlier THE SAME DAY, even across
    separate runs of this script (fixes repeated rows from re-testing)
  - YOLO only runs every Nth frame (config.YOLO_EVERY_N_FRAMES) instead of
    every frame — meaningful FPS improvement, especially over Iriun/network
  - Camera read failures retry a few times instead of instantly ending the
    session (one dropped frame over WiFi shouldn't kill a whole class period)
  - Warns at startup about likely-duplicate student folders (e.g. "Sanjana"
    and "JS Sanjana"), which otherwise silently split one person's training
    data into two separate LBPH classes and hurt recognition accuracy
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from difflib import SequenceMatcher
from ultralytics import YOLO

import config
import database as db

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except Exception as e:
    print("Warning: MediaPipe import failed. Drowsiness detection will be disabled.")
    print(f"Import error: {e}")
    mp_face_mesh = None

# ── EAR Calculation ──────────────────────────────────────────
def calculate_EAR(eye_points, landmarks, w, h):
    p = []
    for idx in eye_points:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        p.append((x, y))
    v1 = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    v2 = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    h1 = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (v1 + v2) / (2.0 * h1)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ── Initialize database ──────────────────────────────────────
db.init_db()
session_id = db.start_session()
print(f"Session {session_id} started.")

# ── Load enrolled students (with normalization + duplicate warning) ──
print("Loading enrolled students...")

known_faces, known_names, known_ids = [], [], []
folder_by_norm_id, folder_by_norm_name = {}, {}   # for duplicate detection

face_detector = cv2.CascadeClassifier(config.get_cascade_path())

if not os.path.isdir(config.STUDENT_PHOTOS_DIR):
    print(f"'{config.STUDENT_PHOTOS_DIR}' folder not found. Enroll students first with enroll_student.py")
    raise SystemExit

for student_folder in os.listdir(config.STUDENT_PHOTOS_DIR):
    folder_path = os.path.join(config.STUDENT_PHOTOS_DIR, student_folder)
    if not os.path.isdir(folder_path):
        continue
    parts = student_folder.split("_", 1)
    if len(parts) < 2:
        print(f"Skipping '{student_folder}' — folder name must be <id>_<name>")
        continue

    student_id   = db.normalize_id(parts[0])
    student_name = db.normalize_name(parts[1])

    # Duplicate detection: same normalized ID under a different folder
    if student_id in folder_by_norm_id and folder_by_norm_id[student_id] != student_folder:
        print(f"WARNING: '{student_folder}' has the same ID as '{folder_by_norm_id[student_id]}' "
              f"— these will be merged as one student ({student_id}). If they're different "
              f"people, fix the ID in one of the folder names.")
    folder_by_norm_id[student_id] = student_folder

    # Duplicate detection: very similar name under a different ID (e.g. "Sanjana" vs "JS Sanjana")
    for existing_name, existing_folder in folder_by_norm_name.items():
        similarity = SequenceMatcher(None, student_name, existing_name).ratio()
        if similarity > 0.6 and existing_folder != student_folder:
            print(f"WARNING: '{student_folder}' looks similar to '{existing_folder}' "
                  f"({student_name} vs {existing_name}). If this is the same person enrolled "
                  f"twice, merge the folders — otherwise recognition accuracy will suffer.")
    folder_by_norm_name[student_name] = student_folder

    for photo_file in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, photo_file))
        if img is None:
            continue
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 5)
        for (x, y, w_f, h_f) in faces:
            face_roi = cv2.resize(gray[y:y+h_f, x:x+w_f], (200, 200))
            known_faces.append(face_roi)
            known_names.append(student_name)
            known_ids.append(student_id)

if len(known_faces) == 0:
    print("No usable training photos found. Run enroll_student.py first.")
    raise SystemExit

unique_names = list(set(known_names))
labels = [unique_names.index(name) for name in known_names]

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(known_faces, np.array(labels))
print(f"Loaded {len(unique_names)} students ({len(known_faces)} training photos).")

# ── Load Models ───────────────────────────────────────────────
print("Loading YOLO model...")
yolo_model = YOLO("yolov8n.pt")
face_mesh = None
if mp_face_mesh is not None:
    try:
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception as e:
        print("Warning: Failed to initialize MediaPipe FaceMesh. Drowsiness detection will be disabled.")
        print(f"Initialization error: {e}")
print("All models loaded. Starting system...")

# ── Tracking variables ────────────────────────────────────────
already_marked_this_run = {}
frame_counter = 0
yolo_frame_count = 0
last_yolo_results = None
last_phone_log = last_drowsy_log = last_engage_log = None
session_start = datetime.now()

# ── Attendance ────────────────────────────────────────────────
def mark_attendance(student_id, student_name):
    student_id = db.normalize_id(student_id)
    if student_id in already_marked_this_run:
        return
    if not config.ALLOW_REMARK_SAME_DAY and db.is_marked_today(student_id):
        already_marked_this_run[student_id] = True   # don't check the DB again every frame
        return

    now = datetime.now()
    class_start = now.replace(hour=config.CLASS_START_HOUR, minute=config.CLASS_START_MINUTE, second=0)
    status = "Late" if now > class_start else "On Time"

    inserted = db.mark_attendance(session_id, student_id, student_name, status)
    if inserted:
        already_marked_this_run[student_id] = True
        print(f"Marked: {student_name} - {status}")

# ── Engagement Score ──────────────────────────────────────────
def calculate_engagement(face_present, phone_detected, drowsy):
    score = 100
    if not face_present: score -= 40
    if phone_detected:   score -= 35
    if drowsy:           score -= 25
    return max(score, 0)

def get_score_color(score):
    if score >= 70: return (0, 255, 0)
    elif score >= 40: return (0, 165, 255)
    else: return (0, 0, 255)

# ── Start camera (with retry on dropped frames) ─────────────────
def open_camera():
    if config.USE_DSHOW:
        return cv2.VideoCapture(config.CAMERA_SOURCE, cv2.CAP_DSHOW)
    return cv2.VideoCapture(config.CAMERA_SOURCE)

cap = open_camera()
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("ClassSentinel running... Press Q to quit and generate report.")

consecutive_failures = 0
MAX_CONSECUTIVE_FAILURES = 30   # ~1 second of dropped frames at 30fps before giving up

while True:
    ret, frame = cap.read()
    if not ret:
        consecutive_failures += 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print("Camera feed lost. Ending session.")
            break
        time.sleep(0.05)
        continue
    consecutive_failures = 0

    h, w, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    now = datetime.now()

    # ── Attendance ────────────────────────────────────────────
    faces = face_detector.detectMultiScale(gray_frame, config.FACE_SCALE_FACTOR, 5, minSize=config.FACE_MIN_SIZE)
    for (x, y, fw, fh) in faces:
        face_roi = cv2.resize(gray_frame[y:y+fh, x:x+fw], (200, 200))
        label, confidence = recognizer.predict(face_roi)
        if config.DEBUG_PRINT_CONFIDENCE:
            print(f"Predicted: {unique_names[label]}, Confidence: {round(confidence, 1)}")
        if confidence < config.CONFIDENCE_THRESHOLD:
            name = unique_names[label]
            uid = known_ids[known_names.index(name)]
            color = (0, 255, 0)
            mark_attendance(uid, name)
        else:
            name, color = "Unknown", (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)
        cv2.putText(frame, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ── Drowsiness ────────────────────────────────────────────
    drowsy = False
    face_present = len(faces) > 0

    if face_mesh is not None:
        mesh_results = face_mesh.process(rgb_frame)
        face_present = mesh_results.multi_face_landmarks is not None
        if face_present:
            for face_landmarks in mesh_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                try:
                    avg_ear = (calculate_EAR(LEFT_EYE, landmarks, w, h) +
                               calculate_EAR(RIGHT_EYE, landmarks, w, h)) / 2.0
                    cv2.putText(frame, f"EAR:{round(avg_ear,2)}", (w-120, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    if avg_ear < config.EAR_THRESHOLD:
                        frame_counter += 1
                        if frame_counter >= config.DROWSY_FRAMES:
                            drowsy = True
                            if last_drowsy_log is None or (now - last_drowsy_log).seconds >= 5:
                                db.log_drowsy_alert(session_id)
                                last_drowsy_log = now
                    else:
                        frame_counter = 0
                    for idx in LEFT_EYE + RIGHT_EYE:
                        cv2.circle(frame, (int(landmarks[idx].x*w), int(landmarks[idx].y*h)), 2, (255,255,0), -1)
                except Exception:
                    pass

    # ── Phone Detection (throttled) ────────────────────────────
    phone_detected = False
    yolo_frame_count += 1
    if yolo_frame_count % config.YOLO_EVERY_N_FRAMES == 0:
        last_yolo_results = yolo_model(frame, verbose=False)
    yolo_results = last_yolo_results

    if yolo_results is not None:
        for result in yolo_results:
            for box in result.boxes:
                if int(box.cls[0]) == 67:
                    phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(frame, "PHONE", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    if last_phone_log is None or (now - last_phone_log).seconds >= 5:
                        db.log_phone_alert(session_id)
                        last_phone_log = now

    # ── Engagement + UI ────────────────────────────────────────
    score = calculate_engagement(face_present, phone_detected, drowsy)
    score_color = get_score_color(score)
    if last_engage_log is None or (now - last_engage_log).seconds >= 10:
        db.log_engagement(session_id, score)
        last_engage_log = now

    cv2.rectangle(frame, (15,15), (310,50), (40,40,40), -1)
    cv2.rectangle(frame, (15,15), (15+int(score*2.9), 50), score_color, -1)
    cv2.putText(frame, f"Engagement: {score}%", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
    cv2.putText(frame, f"Marked : {len(already_marked_this_run)}", (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"Phone  : {'YES' if phone_detected else 'NO'}", (160,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,0,255) if phone_detected else (0,255,0), 2)
    cv2.putText(frame, f"Drowsy : {'YES' if drowsy else 'NO'}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,0,255) if drowsy else (0,255,0), 2)
    if drowsy:
        cv2.putText(frame, "DROWSY ALERT!", (20,135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    if phone_detected:
        cv2.putText(frame, "PHONE DETECTED!", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("ClassSentinel", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
db.end_session(session_id)
db.export_attendance_csv(session_id, config.ATTENDANCE_CSV)
print(f"Session {session_id} closed. Attendance exported to {config.ATTENDANCE_CSV}.")
print("Run 'python backend/app.py' and open http://localhost:5000 to view the dashboard.")