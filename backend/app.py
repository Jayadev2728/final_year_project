"""
backend/app.py
---------------
Flask REST API for ClassSentinel.

Reads from the same SQLite database that detection_system.py writes to,
and serves the frontend dashboard. Run this AFTER detection_system.py
has started (so a session already exists), then open:

    http://localhost:5000

The dashboard polls these endpoints every few seconds to stay live.
"""

import os
import sys
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# Allow "import database" to work when this file is run from backend/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database as db

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

db.init_db()


# ── Frontend ─────────────────────────────────────────────────────
@app.route("/")
def serve_dashboard():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ── Sessions ─────────────────────────────────────────────────────
@app.route("/api/sessions")
def api_sessions():
    return jsonify(db.get_all_sessions())


@app.route("/api/session/latest")
def api_latest_session():
    return jsonify(db.get_latest_session() or {})


# ── Attendance ───────────────────────────────────────────────────
@app.route("/api/attendance/<int:session_id>")
def api_attendance(session_id):
    return jsonify(db.get_attendance(session_id))


# ── Alerts ───────────────────────────────────────────────────────
@app.route("/api/alerts/<int:session_id>")
def api_alerts(session_id):
    return jsonify({
        "phone":  db.get_phone_alerts(session_id),
        "drowsy": db.get_drowsy_alerts(session_id)
    })


# ── Engagement ───────────────────────────────────────────────────
@app.route("/api/engagement/<int:session_id>")
def api_engagement(session_id):
    return jsonify(db.get_engagement_log(session_id))


# ── Summary (top cards) ──────────────────────────────────────────
@app.route("/api/summary/<int:session_id>")
def api_summary(session_id):
    return jsonify(db.get_summary(session_id))


if __name__ == "__main__":
    print("ClassSentinel backend running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
