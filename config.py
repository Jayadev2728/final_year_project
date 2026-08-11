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
CAMERA_SOURCE = 1
USE_DSHOW     = True     # Windows-only camera backend, avoids access-blocked issues

# ── Face recognition (LBPH) ─────────────────────────────────────
CONFIDENCE_THRESHOLD    = 75     # lower = stricter. Raise if enrolled people show "Unknown",
                                  # lower if two different people get mixed up.
DEBUG_PRINT_CONFIDENCE  = False  # True shows live confidence values in the terminal, for tuning
FACE_MIN_SIZE           = (80, 80)   # smallest face (px) to even attempt recognition
FACE_SCALE_FACTOR       = 1.05
FACE_MIN_NEIGHBORS      = 6      # higher = fewer false-positive detections (was hardcoded 5)
FACE_NMS_IOU_THRESHOLD  = 0.3    # overlapping detections above this IoU get collapsed into
                                  # the single strongest one, instead of drawing duplicates
FACE_MIN_CONFIDENCE_SCORE = 0    # Haar's internal confidence score floor. 0 = off (NMS alone
                                  # handles duplicates). Raise this if "ghost" boxes on
                                  # textured backgrounds (patterned curtains/walls) persist —
                                  # set DEBUG_PRINT_CONFIDENCE=True and check main.py's
                                  # terminal output for the score of a ghost detection first.

# ── Drowsiness (EAR) ─────────────────────────────────────────────
EAR_THRESHOLD  = 0.23
DROWSY_SECONDS = 1.5     # continuous eyes-closed time before flagging drowsy.
                          # Time-based rather than frame-count based, because
                          # frame rate varies with how much else is running
                          # (YOLO + MediaPipe + LBPH every frame) — a fixed
                          # frame count can silently mean very different real
                          # durations depending on the machine/camera.
EAR_RESET_TOLERANCE_FRAMES = 3   # how many consecutive above-threshold frames
                                  # are allowed before resetting the drowsy timer.
                                  # Absorbs a single jittery landmark-tracking
                                  # frame instead of resetting on any blip.
DEBUG_PRINT_EAR = False   # True prints live EAR values to the terminal —
                           # use this to find the right EAR_THRESHOLD for your
                           # actual camera/distance instead of guessing.

# ── Face-to-event attribution ──────────────────────────────────────
# When a phone or a closed-eye landmark set is detected, it's matched to
# whichever recognized face is closest to it in the frame. This is a
# proximity heuristic, not a guarantee — if two students sit close
# together, attribution can be wrong. This caps how far away a face can
# be and still count as a match (as a fraction of frame width); anything
# with no recognized face within that range gets logged as unidentified.
FACE_MATCH_MAX_DISTANCE_RATIO = 0.35

# ── Per-student report engagement score ─────────────────────────────
# The report's per-student Engagement column is a simple derived score,
# separate from the single live "Engagement: X%" shown on screen during
# the session (which is a rough whole-frame proxy, not per-person).
# score = 100 - (phone_alerts * PHONE_ALERT_PENALTY) - (drowsy_alerts * DROWSY_ALERT_PENALTY)
PHONE_ALERT_PENALTY  = 15
DROWSY_ALERT_PENALTY = 10

# ── Phone detection (YOLO) ───────────────────────────────────────
YOLO_EVERY_N_FRAMES = 3   # run YOLO every Nth frame instead of every frame — big FPS win,
                           # especially over a network camera stream like Iriun

# ── Attendance ────────────────────────────────────────────────────
CLASS_START_HOUR   = 9
CLASS_START_MINUTE = 0
ALLOW_REMARK_SAME_DAY = False   # if False, re-running main.py the same day won't create
                                 # duplicate attendance rows for someone already marked today

# ── Session duration ─────────────────────────────────────────────
# Set to a number of minutes to auto-stop the session after that long
# (e.g. 60 for a full class period). Set to None to disable — session
# then only ends when you press Q, same as before.
SESSION_DURATION_MINUTES = 60

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
