import cv2
import numpy as np
import mediapipe as mp
import math
import time


# Initialize Mediapipe Pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (shoulder)
    b = np.array(b)  # Midpoint (elbow)
    c = np.array(c)  # Last point (wrist)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Setup capture
cap = cv2.VideoCapture(0)

# Variables to track reps and stage
counter = 0
stage = None  # Either "up" or "down"

# FPS tracking
prev_time = 0

# Initialize pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of shoulder, elbow, and wrist
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle at the elbow
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visual feedback: X-ray vision-style marks
            if angle > 160:  # Elbow is fully extended (arm down)
                stage = "down"
            if angle < 30 and stage == "down":  # Arm is curled (elbow is bent)
                stage = "up"
                counter += 1
                print(f"Bicep Curl Count: {counter}")
                print(f"Angle: {angle} - Stage: {stage}")

            # Display the angle on the frame
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # X-ray vision feedback (green for correct, red for wrong)
            if 30 < angle < 160:
                color = (0, 255, 0)  # Green for correct movement
            else:
                color = (0, 0, 255)  # Red for wrong movement

            # Draw lines between shoulder, elbow, and wrist with the appropriate color
            cv2.line(image, tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                     tuple(np.multiply(elbow, [640, 480]).astype(int)), color, 3)
            cv2.line(image, tuple(np.multiply(elbow, [640, 480]).astype(int)),
                     tuple(np.multiply(wrist, [640, 480]).astype(int)), color, 3)

        except Exception as e:
            print(f"Error: {e}")
            pass

        # Render pose detections on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(image, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the image with the angle and counter
        cv2.imshow('Bicep Curl Detector', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Your while loop and video capture logic
    pass
