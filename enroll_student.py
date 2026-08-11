"""
enroll_student.py
------------------
Webcam-based enrollment. Two fixes vs the previous version:
  1. Only saves a photo if a face was actually detected in it — the old
     version saved whatever was in frame when you hit SPACE, so a bad
     angle/blink could silently contribute zero usable training data.
  2. Warns if the ID or name looks like it might already be enrolled
     (e.g. "Sanjana" vs "JS Sanjana") — enrolling the same person twice
     under different names splits their training data into two separate
     people as far as LBPH is concerned, which hurts everyone's accuracy,
     not just theirs.

Run once per student. Restart main.py afterward — it only loads
student_photos/ once, at startup.
"""

import cv2
import os
from difflib import SequenceMatcher

import config
import database as db

NUM_PHOTOS = 15

def find_possible_duplicates(student_id, student_name):
    if not os.path.isdir(config.STUDENT_PHOTOS_DIR):
        return []
    norm_id, norm_name = db.normalize_id(student_id), db.normalize_name(student_name)
    matches = []
    for folder in os.listdir(config.STUDENT_PHOTOS_DIR):
        parts = folder.split("_", 1)
        if len(parts) < 2:
            continue
        existing_id, existing_name = db.normalize_id(parts[0]), db.normalize_name(parts[1])
        if existing_id == norm_id:
            matches.append(f"{folder} (same ID)")
        elif SequenceMatcher(None, norm_name, existing_name).ratio() > 0.6:
            matches.append(f"{folder} (similar name)")
    return matches

def main():
    student_id   = input("Enter student ID (e.g. 4BD23IS050): ").strip()
    student_name = input("Enter student name (e.g. Jayadev): ").strip()

    if not student_id or not student_name:
        print("ID and name can't be empty.")
        return

    duplicates = find_possible_duplicates(student_id, student_name)
    if duplicates:
        print("\nThis might already be enrolled:")
        for d in duplicates:
            print(f"  - {d}")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != "y":
            print("Cancelled. Delete or rename the existing folder first if you want to re-enroll.")
            return

    norm_id, norm_name = db.normalize_id(student_id), db.normalize_name(student_name)
    save_dir = os.path.join(config.STUDENT_PHOTOS_DIR, f"{norm_id}_{norm_name}")
    os.makedirs(save_dir, exist_ok=True)

    face_detector = cv2.CascadeClassifier(config.get_cascade_path())
    cap = cv2.VideoCapture(config.CAMERA_SOURCE, cv2.CAP_DSHOW) if config.USE_DSHOW else cv2.VideoCapture(config.CAMERA_SOURCE)

    if not cap.isOpened():
        print("Cannot open camera. Check config.py's CAMERA_SOURCE.")
        return

    print(f"\nGet ready {norm_name}! Move your head slightly between shots (left/right/up/down).")
    print(f"Press SPACE to capture ({NUM_PHOTOS} needed), ESC to finish early.\n")

    count = 0
    while count < NUM_PHOTOS:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, config.FACE_SCALE_FACTOR, config.FACE_MIN_NEIGHBORS, minSize=(80, 80))

        display = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_ok = len(faces) > 0

        status_color = (0, 255, 0) if face_ok else (0, 0, 255)
        status_text  = "Face detected - SPACE to capture" if face_ok else "No face detected"
        cv2.putText(display, f"{count}/{NUM_PHOTOS}  {status_text}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.imshow("Enroll Student", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 32 and face_ok:
            cv2.imwrite(os.path.join(save_dir, f"photo{count + 1}.jpg"), frame)
            count += 1
            print(f"Captured {count}/{NUM_PHOTOS}")

    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        print(f"\nNo photos captured. Remove the empty folder at {save_dir} and try again.")
    else:
        print(f"\nDone — {count} photos saved to {save_dir}")
        print("Restart main.py so it loads this student into recognition.")

if __name__ == "__main__":
    main()
