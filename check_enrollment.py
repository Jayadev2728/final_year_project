"""
check_enrollment.py
--------------------
Diagnoses why a student isn't being recognized. Walks student_photos/
exactly the way detection_system.py does, and reports per student:
  - whether the folder name is valid (must be <id>_<name>)
  - how many photos were found
  - how many of those photos actually had a detectable face

A student with 0 detected faces will NEVER be recognized, even though
their folder exists — this is the #1 cause of "added them but it's not
detecting" issues.

Run this from the project root:
    python check_enrollment.py
"""

import cv2
import os
from difflib import SequenceMatcher

import config
import database as db

student_photos_dir = config.STUDENT_PHOTOS_DIR

face_detector = cv2.CascadeClassifier(config.get_cascade_path())

if not os.path.isdir(student_photos_dir):
    print(f"'{student_photos_dir}' folder not found. Run this from your project root.")
    raise SystemExit

print(f"Scanning '{student_photos_dir}'...\n")
print(f"{'Folder':35} {'Valid name?':12} {'Photos':8} {'Faces found':12} {'Status'}")
print("-" * 90)

any_problem = False

for student_folder in sorted(os.listdir(student_photos_dir)):
    folder_path = os.path.join(student_photos_dir, student_folder)
    if not os.path.isdir(folder_path):
        continue

    parts = student_folder.split("_", 1)
    valid_name = len(parts) >= 2

    photo_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    faces_found = 0
    for photo_file in photo_files:
        img = cv2.imread(os.path.join(folder_path, photo_file))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, config.FACE_SCALE_FACTOR, config.FACE_MIN_NEIGHBORS)
        if len(faces) > 0:
            faces_found += 1

    if not valid_name:
        status = "SKIPPED — folder name needs an underscore (id_name)"
        any_problem = True
    elif len(photo_files) == 0:
        status = "NO IMAGE FILES in folder"
        any_problem = True
    elif faces_found == 0:
        status = "0 TRAINING SAMPLES — will never be recognized"
        any_problem = True
    elif faces_found < 3:
        status = "Low sample count — recognition may be unreliable"
        any_problem = True
    else:
        status = "OK"

    print(f"{student_folder:35} {str(valid_name):12} {len(photo_files):<8} {faces_found:<12} {status}")

print("-" * 90)
if any_problem:
    print("\nFix any row above that isn't 'OK', then restart main.py.")
else:
    print("\nAll students loaded cleanly.")

# ── Duplicate detection ──────────────────────────────────────────
print("\nChecking for likely duplicate enrollments...")
folders = [f for f in sorted(os.listdir(student_photos_dir))
           if os.path.isdir(os.path.join(student_photos_dir, f)) and "_" in f]
parsed = [(f, db.normalize_id(f.split("_", 1)[0]), db.normalize_name(f.split("_", 1)[1])) for f in folders]

found_dupe = False
for i in range(len(parsed)):
    for j in range(i + 1, len(parsed)):
        folder_a, id_a, name_a = parsed[i]
        folder_b, id_b, name_b = parsed[j]
        if id_a == id_b:
            print(f"  SAME ID:      '{folder_a}' and '{folder_b}' — will be merged as one student.")
            found_dupe = True
        elif SequenceMatcher(None, name_a, name_b).ratio() > 0.6:
            print(f"  SIMILAR NAME: '{folder_a}' and '{folder_b}' — likely the same person enrolled twice.")
            found_dupe = True

if found_dupe:
    print("\nIf any of the above are really the same person, merge their photos into ONE")
    print("folder (same ID, same name) and delete the duplicate — otherwise their face")
    print("data is split across two people and recognition accuracy suffers for everyone.")
else:
    print("  None found.")
