from flask import Flask, render_template, Response, request, jsonify
import cv2
import imutils
import numpy as np
import os
import time
import threading
import winsound

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize tracker
TrDict = {
    'csrt': cv2.legacy.TrackerCSRT_create,
    'kcf': cv2.legacy.TrackerKCF_create,
    'boosting': cv2.legacy.TrackerBoosting_create,
    'mil': cv2.legacy.TrackerMIL_create,
    'tld': cv2.legacy.TrackerTLD_create,
    'medianflow': cv2.legacy.TrackerMedianFlow_create,
    'mosse': cv2.legacy.TrackerMOSSE_create
}
tracker = None
video_path = None
last_seen = time.time()
missing_threshold = 2  # Seconds before missing alert
object_lost = False
obstruction_count = 0  # Count obstruction occurrences
obstruction_start_time = 0  # Time when obstruction started

# Store tracking data
tracking_data = {
    "object_count": 0,
    "obstruction_time": None,
    "current_time": None,
    "status": "Idle"
}

# Function to play beep sound in parallel
def play_beep(frequency, duration, count):
    def sound_thread():
        for _ in range(count):
            winsound.Beep(frequency, duration)
            time.sleep(0.1)
    threading.Thread(target=sound_thread, daemon=True).start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_path
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    video_path = file_path
    tracking_data["status"] = "File Tracking"
    return jsonify({'success': True})

@app.route('/start_file_tracking')
def start_file_tracking():
    return Response(track_objects(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_live_tracking')
def start_live_tracking():
    tracking_data["status"] = "Live Tracking"
    return Response(track_objects(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/tracking_data', methods=['GET'])
def get_tracking_data():
    tracking_data["current_time"] = time.strftime("%I:%M:%S %p")
    return jsonify(tracking_data)

@app.route('/api/update_obstruction', methods=['POST'])
def update_obstruction():
    global obstruction_count, obstruction_start_time
    data = request.get_json()
    obstruction_count = data.get('obstruction_count', obstruction_count)
    if obstruction_count > 0 and obstruction_start_time == 0:
        obstruction_start_time = time.time()
    elif obstruction_count == 0:
        obstruction_start_time = 0
    tracking_data.update({
        "object_count": obstruction_count,
        "obstruction_time": time.strftime("%I:%M:%S %p") if obstruction_start_time else None,
        "status": data.get('status', tracking_data["status"])
    })
    return jsonify({'success': True})

def track_objects(source):
    global tracker, last_seen, object_lost, obstruction_count, obstruction_start_time
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    if not ret:
        return
    frame = imutils.resize(frame, width=600)
    tracker = TrDict['mosse']()
    bb = cv2.selectROI('Frame', frame, False)
    tracker.init(frame, bb)
    prev_x, prev_y = bb[0], bb[1]
    ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        success, box = tracker.update(frame)

        if success:
            (x, y, w, h) = [int(a) for a in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 0), 2)
            movement_x = abs(x - prev_x) / w
            movement_y = abs(y - prev_y) / h

            if movement_x > 0.3 or movement_y > 0.3:
                play_beep(2500, 500, 3)

            diff = cv2.absdiff(ref_gray[y:y+h, x:x+w], gray[y:y+h, x:x+w])
            obstruction_level = np.mean(diff)

            if obstruction_level > 10:
                if obstruction_start_time == 0:
                    obstruction_start_time = time.time()
                obstruction_count += 1
                obstruction_duration = time.time() - obstruction_start_time

                if obstruction_duration > 10:
                    play_beep(2000, 500, 1)
                    obstruction_start_time = 0
                    obstruction_count = 0
            else:
                obstruction_start_time = 0

            last_seen = time.time()
            object_lost = False
            prev_x, prev_y = x, y
        else:
            missing_time = time.time() - last_seen
            if missing_time >= missing_threshold:
                object_lost = True
                play_beep(1500, 300, 20)

        if obstruction_start_time != 0:
            obstruction_count+=1
            cv2.putText(frame, f"Obstruction : {int(time.time() - obstruction_start_time)}s", (50, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')