"""
config.py
----------
Single place for every tunable value in the project. The old repo had
CONFIDENCE_THRESHOLD hardcoded differently in main.py (75) and
take_attendence.py (70) — that kind of drift is exactly what causes
"it worked yesterday" bugs. Change values here, not inside main.py.
"""

# ── Camera ───────────────────────────────────────────────────────
# 0 = laptop webcam. For Iriun (phone), use the index camera.py told you.
CAMERA_SOURCE = 0
USE_DSHOW     = True     # Windows-only camera backend, avoids access-blocked issues

# ── Face recognition (LBPH) ─────────────────────────────────────
CONFIDENCE_THRESHOLD    = 75     # lower = stricter. Raise if enrolled people show "Unknown",
                                  # lower if two different people get mixed up.
DEBUG_PRINT_CONFIDENCE  = False  # True shows live confidence values in the terminal, for tuning
FACE_MIN_SIZE           = (80, 80)   # smallest face (px) to even attempt recognition
FACE_SCALE_FACTOR       = 1.05

# ── Drowsiness (EAR) ─────────────────────────────────────────────
EAR_THRESHOLD = 0.23
DROWSY_FRAMES = 20     # consecutive low-EAR frames before flagging drowsy

# ── Phone detection (YOLO) ───────────────────────────────────────
YOLO_EVERY_N_FRAMES = 3   # run YOLO every Nth frame instead of every frame — big FPS win,
                           # especially over a network camera stream like Iriun

# ── Attendance ────────────────────────────────────────────────────
CLASS_START_HOUR   = 9
CLASS_START_MINUTE = 0
ALLOW_REMARK_SAME_DAY = False   # if False, re-running main.py the same day won't create
                                 # duplicate attendance rows for someone already marked today

# ── Paths ────────────────────────────────────────────────────────
STUDENT_PHOTOS_DIR = "student_photos"
ATTENDANCE_CSV      = "attendance.csv"

def get_cascade_path():
    """Finds the Haar cascade file, preferring the copy bundled in this
    project over cv2's install path. This avoids a real issue where
    pip/conda installs can put cv2's data files somewhere unexpected
    (e.g. a different Python install than the one actually running),
    causing 'Can't open file' errors that have nothing to do with your code."""
    import os
    import cv2

    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
    if os.path.exists(local_path):
        return local_path

    fallback_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if os.path.exists(fallback_path):
        return fallback_path

    raise FileNotFoundError(
        "Could not find haarcascade_frontalface_default.xml locally or in cv2's install. "
        "Make sure haarcascade_frontalface_default.xml is in the project root."
    )