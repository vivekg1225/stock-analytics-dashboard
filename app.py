from flask import Flask, render_template
from flask_socketio import SocketIO
import json
import os
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stocksecret'
socketio = SocketIO(app, cors_allowed_origins="*")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "results.json")
BATCH_FILE   = os.path.join(BASE_DIR, "batch_results.json")

last_data   = []
start_time  = time.time()


# File readers 

def read_results():
    """Read latest streaming results from Spark output file."""
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
    except Exception as e:
        print(f"Error reading results.json: {e}")
    return []


def read_batch():
    """Read latest batch analytics from spark_batch.py output."""
    try:
        if os.path.exists(BATCH_FILE):
            with open(BATCH_FILE, "r") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
    except Exception as e:
        print(f"Error reading batch_results.json: {e}")
    return {}


# Payload builder 

def build_payload(stream_data, batch_data):
    """
    Merges streaming stats and batch analytics into a single payload
    for the frontend. Batch data is optional — frontend handles absence.
    """
    total_ticks  = sum(row.get("total_ticks", 0) for row in stream_data)
    anomaly_count = sum(1 for row in stream_data if row.get("is_anomaly", False))
    uptime       = round(time.time() - start_time)

    return {
        # Stream data 
        "stocks"         : stream_data,
        "total_messages" : total_ticks,
        "uptime_seconds" : uptime,
        "active_stocks"  : len(stream_data),
        "anomaly_count"  : anomaly_count,

        # Batch data (may be empty dict if batch hasn't run yet)
        "batch"          : batch_data,
    }


# Background push thread

def push_updates():
    """Background thread: push merged results to all clients every 2s."""
    global last_data
    while True:
        time.sleep(2)
        stream_data = read_results()
        if stream_data:
            batch_data  = read_batch()
            payload     = build_payload(stream_data, batch_data)
            socketio.emit("update", payload)
            last_data = stream_data


# Routes

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    """REST endpoint — streaming stats only."""
    stream_data = read_results()
    batch_data  = read_batch()
    return build_payload(stream_data, batch_data)


@app.route("/api/batch")
def api_batch():
    """REST endpoint — raw batch analytics JSON."""
    return read_batch() or {"status": "no_batch_data"}


# Socket events 

@socketio.on("connect")
def on_connect():
    print("Client connected")
    stream_data = read_results()
    if stream_data:
        batch_data = read_batch()
        payload    = build_payload(stream_data, batch_data)
        socketio.emit("update", payload)


# Entry point

if __name__ == "__main__":
    thread = threading.Thread(target=push_updates, daemon=True)
    thread.start()
    print("Flask server running at http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)