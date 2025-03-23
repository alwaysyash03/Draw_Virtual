import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Initialize variables to store previous finger position
prev_x, prev_y = None, None

# Create a canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the tip of the index finger (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x = int(index_finger_tip.x * w)
            y = int(index_finger_tip.y * h)

            # Draw on the canvas
            if prev_x and prev_y:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)

            # Update previous finger position
            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    # Combine the frame and the canvas
    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Show the frame
    cv2.imshow('Virtual Drawing', combined_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
