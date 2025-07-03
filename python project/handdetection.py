import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import threading

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


SOUND_FILE = "alert.wav"  # Ensure you have an alert.wav file in the same directory

def play_sound():
    try:
        playsound(SOUND_FILE)
    except Exception as e:
        print(f"Error playing sound: {e}")

# Variables for hand raise detection
last_sound_time = 0
SOUND_COOLDOWN = 2  # Seconds between sound triggers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    # Initialize status message
    status = "No Hand"
    hand_raised = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get wrist landmark (landmark 0)
            wrist = hand_landmarks.landmark[0]
            frame_height = frame.shape[0]

            # Check if hand is raised (wrist y-coordinate is in upper half of frame)
            if wrist.y < 0.5:
                hand_raised = True
                status = "Hand Raised!"
            else:
                status = "Hand Detected"

            # Trigger sound for raised hand with cooldown
            current_time = time.time()
            if hand_raised and (current_time - last_sound_time) > SOUND_COOLDOWN:
                threading.Thread(target=play_sound, daemon=True).start()
                last_sound_time = current_time

    # Display status message
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if status != "No Hand" else (0, 0, 255), 2)

    
    cv2.imshow("Hand Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
