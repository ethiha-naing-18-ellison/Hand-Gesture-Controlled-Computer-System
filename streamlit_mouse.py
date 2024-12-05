import cv2
import numpy as np
import mediapipe as mp
import time
import HandTrackingModule as htm
import streamlit as st
import os

try:
    if "DISPLAY" in os.environ:
        import pyautogui
    else:
        print("Running in headless mode. Skipping pyautogui.")
except ImportError:
    print("PyAutoGUI is not available in this environment.")

# Streamlit page configuration
st.title("Hand Tracking Mouse Control")
st.sidebar.header("Controls")

# Initialize control variables
start_app = st.sidebar.button("START", key="start_button")
stop_app = st.sidebar.button("STOP", key="stop_button")

# Sidebar settings
frameR = st.sidebar.slider("Frame Reduction", 50, 200, 100)  # Adjustable frame reduction
smoothening = st.sidebar.slider("Smoothening", 1, 10, 7)  # Adjustable smoothening factor

# Streamlit placeholder for video feed
frame_placeholder = st.empty()

# Camera and Hand Tracking variables
wCam, hCam = 640, 480
cap = None  # Camera capture object
pTime = 0

if start_app:
    st.write("Application Started!")
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    # Initialize variables
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    detector = htm.handDetector(maxHands=1)
    wScr, hScr = pyautogui.size()

    while True:
        # Check if the STOP button is pressed
        if stop_app:
            st.write("Application Stopped!")
            cap.release()
            cv2.destroyAllWindows()
            break

        # Read camera frame
        success, img = cap.read()
        if not success:
            st.warning("Failed to access the camera.")
            break

        # Process hand landmarks
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # Check which fingers are up
            fingers = detector.fingersUp()

            # Draw rectangle for frame reduction
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            # Moving mode
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # Smoothen values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # Move mouse
                pyautogui.moveTo(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY



        # Calculate and display frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Display camera feed in Streamlit
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img, channels="RGB")

if stop_app and cap is not None:
    st.write("Application Stopped!")
    cap.release()
    cv2.destroyAllWindows()
