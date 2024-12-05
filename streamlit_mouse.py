import cv2
import numpy as np
import mediapipe as mp
import time
import HandTrackingModule as htm
import streamlit as st
from pynput.mouse import Controller, Button

# Initialize the mouse controller
mouse = Controller()

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
    wScr, hScr = st.sidebar.number_input("Screen Width", value=1920), st.sidebar.number_input("Screen Height", value=1080)

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
            x1, y1 = lmList[8][1:]  # Index finger tip
            x2, y2 = lmList[12][1:]  # Middle finger tip

            # Check which fingers are up
            fingers = detector.fingersUp()

            # Draw rectangle for frame reduction
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

            # Moving mode (Index finger up, middle finger down)
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # Smoothen values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # Move mouse using pynput
                mouse.position = (wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # Left click (Index and middle fingers up)
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                mouse.click(Button.left, 1)
                st.write("Left click performed.")
                time.sleep(0.2)  # Prevent multiple rapid clicks

            # Right click (Index, middle, and ring fingers up)
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
                mouse.click(Button.right, 1)
                st.write("Right click performed.")
                time.sleep(0.2)  # Prevent multiple rapid clicks

            # Double click (Index, middle, ring, and pinky fingers up)
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                mouse.click(Button.left, 2)
                st.write("Double click performed.")
                time.sleep(0.5)  # Prevent multiple double clicks

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
