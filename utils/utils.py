import mediapipe as mp
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

def generate_df_keynames():
    mp_holistic = mp.solutions.holistic
    
    coords = ["x", "y", "z"]
    keypoint_names = []
    keypoint_names.append("frame")
    keypoint_names.append("class")

    count = 0
    for landmark in mp_holistic.HandLandmark:
        for coord in coords:
            keypoint_names.append("R"+str(landmark) + "_" + str(count) + "_" + coord)
        count+=1

    count = 0
    for landmark in mp_holistic.HandLandmark:
        for coord in coords:
            keypoint_names.append("L"+str(landmark) + "_" + str(count) + "_" + coord)
        count+=1

    count = 0
    for landmark in mp_holistic.PoseLandmark:
        for coord in coords:
            keypoint_names.append(str(landmark) + "_" + str(count) + "_" + coord)
        count+=1
        if count == 15:
            break
    
    df = pd.DataFrame([], columns= keypoint_names)

    return df


def generate_empty_keypoints():
    hand_keypoints = 21 * 3
    pose_keypoints = 15 * 3
    hand_empty_keypoints = np.empty(hand_keypoints)
    pose_empty_keypoints = np.empty(pose_keypoints)
    hand_empty_keypoints[:] = 0.0 #np.nan
    pose_empty_keypoints[:] = 0.0 #np.nan
    hand_empty_keypoints = list(hand_empty_keypoints)
    pose_empty_keypoints = list(pose_empty_keypoints)
    return hand_empty_keypoints, pose_empty_keypoints


def process_video_keypoints(video_path, gesture_class):
    print(f"Procesando video {video_path}")
    df = generate_df_keynames()
    hand_empty_keypoints, pose_empty_keypoints = generate_empty_keypoints()
    mp_holistic = mp.solutions.holistic
    HEIGHT = 600
    WIDTH  = 900
    frame_number = 0
    cap = cv2.VideoCapture(video_path)
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        while cap.isOpened():
            # Leer sequencia
            ret, frame = cap.read()
            if ret == True:
                frame_number += 1
                # Redimensionar sequencia
                frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
                # Cambiar  color de BGR a RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                # Deteccion puntos de referencia
                results = holistic.process(frame)
                try:
                    # Añadir coordenadas a la lista
                    row = []
                    # El nombre de las palabras es añadida a la lista
                    row.append(frame_number)
                    row.append(gesture_class)
                    
                    ## Detectar puntos mano derecha (rhand)
                    if results.right_hand_landmarks:
                        for landmark in results.right_hand_landmarks.landmark:
                            row.append(landmark.x)
                            row.append(landmark.y)
                            row.append(landmark.z)    
                    else:
                        row.extend(hand_empty_keypoints)
                            
                    ## Detectar puntos mano iquierda (lhand)
                    if results.left_hand_landmarks:
                        for landmark in results.left_hand_landmarks.landmark:
                            row.append(landmark.x)
                            row.append(landmark.y)
                            row.append(landmark.z)    
                    else:
                        row.extend(hand_empty_keypoints)
                    
                    ## Detectar puntos cuerpo (pose)
                    if results.pose_landmarks:
                        count = 0
                        for landmark in results.pose_landmarks.landmark:
                            row.append(landmark.x)
                            row.append(landmark.y)
                            row.append(landmark.z)
                            count += 1
                            if count == 15:
                                break    
                    else:
                        row.extend(pose_empty_keypoints)
                    df.loc[len(df)] = row
                except Exception as e:
                    print(e)
            else:
                break
        cap.release()
        print("Video procesado")
        return df
        