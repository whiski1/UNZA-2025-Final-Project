import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.8, max_num_faces=1, refine_landmarks=True)

# Open webcam
cap = cv2.VideoCapture(0)

def get_head_center(landmarks):
    """ Compute head center as the midpoint between the eyes """
    left_eye = np.array(landmarks[33])  # Left eye corner
    right_eye = np.array(landmarks[263])  # Right eye corner
    return ((left_eye + right_eye) / 2).astype(int)

def get_eye_gaze(landmarks):
    """ Estimate eye gaze direction """
    left_pupil = np.array(landmarks[468])  # Left pupil
    right_pupil = np.array(landmarks[473])  # Right pupil
    left_eye_center = np.array(landmarks[33])  # Left eye
    right_eye_center = np.array(landmarks[263])  # Right eye

    # Compute horizontal gaze direction
    left_gaze = left_pupil[0] - left_eye_center[0]
    right_gaze = right_pupil[0] - right_eye_center[0]

    # Compute vertical gaze direction
    left_gaze_y = left_pupil[1] - left_eye_center[1]
    right_gaze_y = right_pupil[1] - right_eye_center[1]

    gaze_x = (left_gaze + right_gaze) / 2
    gaze_y = (left_gaze_y + right_gaze_y) / 2

    # Compute angle of gaze
    gaze_angle = np.degrees(np.arctan2(gaze_y, gaze_x))
    return gaze_angle, gaze_x, gaze_y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0])) for p in face_landmarks.landmark]

            # Get head center
            head_center = get_head_center(landmarks)

            # Compute gaze direction
            gaze_angle, gaze_x, gaze_y = get_eye_gaze(landmarks)

            # Define gaze endpoint (extend line)
            scalex = 50  # Line length
            scaley = 100  # Line length
            end_point = (int(head_center[0] + gaze_x * scalex), int(head_center[1] + gaze_y * scaley))

            # Draw head center
            cv2.circle(frame, tuple(head_center), 5, (0, 255, 0), -1)

            # Draw gaze line
            cv2.line(frame, tuple(head_center), end_point, (0, 255, 255), 2)

            # Display gaze angle
            cv2.putText(frame, f"Gaze Angle: {gaze_angle:.2f}°", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Head Pose & Eye Gaze Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
