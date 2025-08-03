import cv2
import mediapipe as mp
import math
import numpy as np
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Camera cant be opened, make sure your camera is working!")

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize models
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Drawing specs
hand_landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
hand_connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)

# Thresholds
MOUTH_OPEN_THRESHOLD = 0.02
EYE_CLOSED_THRESHOLD = 1

# Landmark indices
finger_tips = [4, 8, 12, 16, 20]

# Waktu awal untuk FPS
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    ih, iw, _ = frame.shape

    # Hands
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                hand_landmark_drawing_spec,
                hand_connection_drawing_spec)

            landmarks = hand_landmarks.landmark
            fingers = []

            for tip in finger_tips:
                if tip == 4:
                    if landmarks[tip].x < landmarks[tip - 1].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if landmarks[tip].y < landmarks[tip - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

            count = sum(fingers)
            cv2.putText(frame, f'Fingers: {count}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Face
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

            landmarks = face_landmarks.landmark

            # Mouth
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            mouth_open = lower_lip.y - upper_lip.y

            # Eyes
            left_eye_closed = landmarks[159].y - landmarks[145].y
            right_eye_closed = landmarks[386].y - landmarks[374].y

            # Gaze direction
            left_eye_x = landmarks[468].x
            left_iris_x = landmarks[473].x
            right_eye_x = landmarks[473].x
            right_iris_x = landmarks[468].x

            left_gaze_direction = "Center"
            if left_iris_x < left_eye_x - 0.05:
                left_gaze_direction = "Left"
            elif left_iris_x > left_eye_x + 0.05:
                left_gaze_direction = "Right"

            right_gaze_direction = "Center"
            if right_iris_x < right_eye_x - 0.05:
                right_gaze_direction = "Left"
            elif right_iris_x > right_eye_x + 0.05:
                right_gaze_direction = "Right"

            gaze_direction = "Looking "
            if left_gaze_direction == right_gaze_direction:
                gaze_direction += left_gaze_direction
            else:
                gaze_direction += "Around"

            # Expressions
            expression_text = ""
            if mouth_open > MOUTH_OPEN_THRESHOLD:
                expression_text += "Mouth Open "
            if left_eye_closed < EYE_CLOSED_THRESHOLD or right_eye_closed < EYE_CLOSED_THRESHOLD:
                expression_text += "Eyes Open "

            # Draw text
            y_pos = 100
            if expression_text:
                cv2.putText(frame, f'Expressions: {expression_text}', (30, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                y_pos += 40

            cv2.putText(frame, gaze_direction, (30, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1]-150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Title
    cv2.putText(frame, "Advanced Face & Hand Tracking", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Show
    cv2.imshow("Advanced Face & Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
