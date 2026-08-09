# ClassSentinel

Intelligent classroom monitoring system — face-recognition attendance,
phone-usage detection (YOLOv8), and drowsiness detection (MediaPipe EAR),
with a live web dashboard.

## Design decisions (and why)

**Full stack: yes, added SQLite + Flask + a web dashboard.**
The CV pipeline was already solid. What was missing for a "system"-framed
capstone was a way to actually view results without reading raw CSVs —
and VTU-style report rubrics generally expect distinct frontend/backend/
database layers. SQLite over PostgreSQL/MySQL because it needs zero setup
(no server to install or explain in a viva), and Flask over FastAPI
because it's simpler to explain live even though `check.py` in the repo
shows FastAPI was considered at one point.

**Camera: kept flexible, no hardware purchase required.**
Iriun (phone-as-webcam) already works and is free. `config.py` holds a
single `CAMERA_SOURCE` value, so switching between laptop webcam / Iriun /
a future USB webcam is a one-line change, not a rewrite. If buying real
hardware later, a budget wide-FOV webcam (e.g. EMEET C960, ~₹2,800, 90°
FOV) matters more than 4K resolution — both YOLOv8 and LBPH downscale
their input internally, so extra resolution mostly helps with faces that
are already far from the camera, not general accuracy.

## What was actually fixed (found by inspecting the real repo data)

Your `attendance.csv` had the same student logged under different names/IDs:
`4bd23IS050` vs `4BD23IS050` (Jayadev), `JS Sanjana` vs `Sanjana`,
`Prashanth KL` vs `Prashanth`. Two real problems came from this:

1. **Attendance analytics were wrong** — the same person counted as 2-3
   different students in reports.
2. **Recognition accuracy was actively hurt** — if the same person is
   enrolled under two different folder names, LBPH treats them as two
   different people, splitting their training photos and weakening both.

Fixes:
- `database.py` normalizes every ID (uppercase) and name (capitalized,
  with initials like "KL"/"JS" preserved correctly) before writing, so
  `4bd23IS050` and `4BD23IS050` are always the same student.
- `enroll_student.py` and `check_enrollment.py` now warn you if a new
  enrollment looks like a duplicate of an existing one (same ID, or a
  similar name), before you end up with two folders for one person.
- `main.py` prints the same warning at startup if it finds this in your
  existing `student_photos/` folders — check the terminal the first time
  you run it after copying these files in.

Other reliability fixes, also based on things visible in the real data/repo:
- **Re-running `main.py` no longer creates duplicate attendance rows** —
  it checks whether someone was already marked *today* (any session), not
  just in the current run. Your old CSV had the same person marked 8+
  times in one day from repeated test runs.
- **YOLO now runs every 3rd frame** instead of every frame — real FPS
  improvement, especially over Iriun's WiFi stream.
- **Camera read failures retry** for ~1 second instead of instantly
  ending the session — one dropped WiFi frame from a phone camera
  shouldn't kill a whole class period.
- **One `CONFIDENCE_THRESHOLD` in `config.py`** instead of different
  hardcoded values in `main.py` (75) and `take_attendence.py` (70).

## Architecture

```
┌──────────────┐   writes    ┌──────────────────┐   reads    ┌─────────────────┐
│   main.py     │ ──────────→ │  classentinel.db  │ ←───────── │  backend/app.py  │
│ (CV pipeline) │             │     (SQLite)        │            │  (Flask REST API) │
└──────────────┘             └──────────────────┘            └────────┬────────┘
                                                                        │ serves
                                                                        ▼
                                                          ┌────────────────────────┐
                                                          │  frontend/index.html     │
                                                          │  (dashboard, polls every  │
                                                          │   4s for live updates)     │
                                                          └────────────────────────┘
```

`main.py` also exports `attendance.csv` at the end of each session (via
`database.export_attendance_csv`), so anything expecting the old CSV
format still works — the database is just the single source of truth now,
instead of the CSV being written to directly and risking drift (this repo
had both `attendance.csv` and an empty stray `attendence.csv` — a typo
that's easy to make when writing CSV rows by hand in multiple places).

## Folder structure

```
final_year_project/
├── config.py              # every tunable value — camera source, thresholds, etc.
├── database.py            # SQLite schema + all reads/writes (shared by main.py and the API)
├── main.py                 # the CV pipeline — run this to start a session
├── enroll_student.py        # webcam enrollment, with duplicate detection
├── check_enrollment.py       # diagnostic: verifies photos have detectable faces, flags dupes
├── camera.py                  # visual picker for laptop webcam vs Iriun camera index
├── requirements.txt
├── student_photos/             # enrolled students — <id>_<name>/*.jpg
├── backend/
│   └── app.py                    # Flask REST API + serves the dashboard
└── frontend/
    └── index.html                  # dashboard (single file, no build step)
```

**Old prototype files** (`face_detection.py`, `phone_detection.py`,
`drowsiness_detection.py`, `engagement_score.py`, `ear_test.py`,
`take_attendence.py`, `test_camera.py`, `check.py`) were your Phase 1
standalone module tests before integrating everything into `main.py` —
worth keeping for the report ("we built and tested each module
independently before integration") but not part of the running system
anymore. Move them out of the way so they don't get run by mistake:

```bash
mkdir legacy
git mv face_detection.py phone_detection.py drowsiness_detection.py engagement_score.py ear_test.py take_attendence.py test_camera.py check.py legacy/
git rm attendence.csv          # empty, unused typo file
git add config.py database.py main.py enroll_student.py check_enrollment.py camera.py requirements.txt backend frontend
git commit -m "Add SQLite + Flask dashboard, fix duplicate-student data issue"
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate              # Windows
pip install -r requirements.txt
```

Put enrolled students' photos under `student_photos/`, one folder per
student, named `<id>_<name>` — or just use `enroll_student.py` (below),
which creates these for you and checks for duplicates first.

## Running it (3 terminals)

**1. Enroll students (skip if `student_photos/` is already populated):**
```bash
python enroll_student.py
```
Repeat once per student. Then sanity-check everything loaded correctly:
```bash
python check_enrollment.py
```

**2. If using a phone camera, find its index:**
```bash
python camera.py
```
Press **N**/**Q** to identify it, then set it in `config.py`:
```python
CAMERA_SOURCE = 1   # whatever index you confirmed
```

**3. Run the detection pipeline:**
```bash
python main.py
```
Watch the terminal for `Marked: <name> - <status>` and any duplicate
warnings. Press **Q** in the video window to end the session.

**4. Run the backend and view the dashboard:**
```bash
cd backend
python app.py
```
Open `http://localhost:5000` — the session dropdown lets you switch
between the live session and past ones.

## Tuning recognition accuracy

If someone shows "Unknown" or gets misidentified, open `config.py`:
```python
DEBUG_PRINT_CONFIDENCE = True   # prints real confidence values to the terminal
```
Run `main.py` again, note the actual numbers for that person, then adjust:
```python
CONFIDENCE_THRESHOLD = 75   # lower = stricter, higher = more lenient
```
Set `DEBUG_PRINT_CONFIDENCE` back to `False` once you're happy with it —
it's noisy in normal use.

## For your report

- **Frontend:** HTML5, CSS3, vanilla JavaScript, Chart.js (CDN)
- **Backend:** Python, Flask, Flask-CORS (REST API)
- **Database:** SQLite
- **CV/ML:** OpenCV (Haar Cascade + LBPH), Ultralytics YOLOv8, MediaPipe
  Face Mesh, NumPy
- **Camera:** phone via Iriun (virtual webcam driver) for the working
  demo; budget wide-FOV USB webcam recommended if hardware is purchased
