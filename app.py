from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import time
from pynput.mouse import Controller, Button
import HandTrackingModule as htm
import threading

app = Flask(__name__)

# Initialize the mouse controller
mouse = Controller()

# Hand Tracking and Camera Variables
cap = None
detector = htm.handDetector(maxHands=1)
running = False  # To control the start and stop functionality
thread = None  # Thread for the video feed loop

# Smoothing variables
frameR = 100  # Frame Reduction
smoothening = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0
wScr, hScr = 1920, 1080  # Default screen size (adjustable in the frontend)


def video_loop():
    """Threaded function for the video feed."""
    global cap, running, plocX, plocY, clocX, clocY

    # Initialize the camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Camera width
    cap.set(4, 480)  # Camera height

    # Initialize the position variables
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    while running:
        success, img = cap.read()
        if not success or not running:
            break

        # Process hand landmarks
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]  # Index finger tip
            x2, y2 = lmList[12][1:]  # Middle finger tip

            # Check which fingers are up
            fingers = detector.fingersUp()

            # Draw rectangle for frame reduction
            cv2.rectangle(img, (frameR, frameR), (640 - frameR, 480 - frameR), (255, 0, 255), 2)

            # Moving mode (Index finger up, middle finger down)
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, 640 - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, 480 - frameR), (0, hScr))

                # Smoothen values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # Move mouse using pynput
                mouse.position = (wScr - clocX, clocY)
                plocX, plocY = clocX, clocY

            # Left click (Index and middle fingers up)
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                mouse.click(Button.left, 1)
                time.sleep(0.2)

            # Right click (Index, middle, and ring fingers up)
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
                mouse.click(Button.right, 1)
                time.sleep(0.2)

            # Double click (Index, middle, ring, and pinky fingers up)
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                mouse.click(Button.left, 2)
                time.sleep(0.5)

    # Release the camera when stopped
    if cap:
        cap.release()
        cap = None


def generate_frames():
    """Generate frames for the video feed."""
    while running:
        if cap:
            success, img = cap.read()
            if not success:
                break

            # Encode the image as a JPEG and return as bytes
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Home page with the video feed and controls."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    if running:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({"message": "Camera is not running"}), 404


@app.route('/start', methods=['POST'])
def start_camera():
    """Start the camera."""
    global running, thread
    if not running:
        running = True
        thread = threading.Thread(target=video_loop)
        thread.start()
        return jsonify({"message": "Camera started"})
    return jsonify({"message": "Camera is already running"})


@app.route('/stop', methods=['POST'])
def stop_camera():
    """Stop the camera."""
    global running, thread
    if running:
        running = False
        if thread:
            thread.join()
        return jsonify({"message": "Camera stopped"})
    return jsonify({"message": "Camera is not running"})


@app.route('/update_settings', methods=['POST'])
def update_settings():
    """Update settings like screen width and height."""
    global wScr, hScr
    data = request.json
    wScr = int(data.get("screenWidth", wScr))
    hScr = int(data.get("screenHeight", hScr))
    return jsonify({"message": "Settings updated"})


if __name__ == '__main__':
    app.run(debug=True)
