# Run MediaPipe Holistic and draw pose landmarks.
import mediapipe as mp
import cv2
import pandas as pd
from pathlib import Path
import sys

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

len_pose_keypoints = 33
len_hand_keypoints = 21
len_face_keypoints = 468

pose = mp_holistic.PoseLandmark._member_names_
left_hand = ['LEFT_' + item for item in mp_holistic.HandLandmark._member_names_]
right_hand = ['RIGHT_' + item for item in mp_holistic.HandLandmark._member_names_]
face_keypoints = ['FACE_' + str(n) for n in range(1, len_face_keypoints+1)]
keypoint_names = ['frame'] + pose + face_keypoints + left_hand + right_hand
df = pd.DataFrame([], columns= keypoint_names)

#cap = cv2.VideoCapture('gestures/hola.mp4')
cap = cv2.VideoCapture(0)
frame = 0
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        # Convert the BGR image to RGB and process it with MediaPipe Pose.
        image = cv2.flip(image, 1)
        results = holistic.process(image)
        
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        
        try:
            pose = list(results.pose_landmarks.landmark)
            face = list(results.face_landmarks.landmark)
            left_hand = list(results.left_hand_landmarks.landmark)
            right_hand = list(results.right_hand_landmarks.landmark)
            keypoints = [frame] + pose + face + left_hand + right_hand
            df.loc[str(frame)] = keypoints
        except AttributeError:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_tb.tb_lineno, frame)
            
            if results.pose_landmarks is None:
                print('No pose')
                pose = [None] * len_pose_keypoints
                
            if results.face_landmarks is None:
                print('No face')
                face = [None] * len_face_keypoints
                
            if results.left_hand_landmarks is None:
                print('No left hand')
                left_hand = [None] * len_hand_keypoints
                
            if results.right_hand_landmarks is None:
                print('No right hand')
                right_hand = [None] * len_hand_keypoints
                
            keypoints = [frame] + pose + face + left_hand + right_hand
            df.loc[str(frame)] = keypoints
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_tb.tb_lineno)
        
        cv2.imshow('MediaPipe Holistic', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        frame += 1
    cap.release()
        
df.to_csv('hola.csv')
cap.release()