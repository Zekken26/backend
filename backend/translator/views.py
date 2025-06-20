import os
import numpy as np
import cv2
import tensorflow as tf
import mediapipe as mp
import gdown

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser

# Define model path and download URL
MODEL_PATH = settings.MODEL_PATH
GOOGLE_DRIVE_FILE_ID = '1jF-UFZfHnpH-EASf3rgQ0x8XpLZayNkW'  
DOWNLOAD_URL = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'

# Download the model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
actions = ['hello', 'ikinagagalak kong makilala ka', 'magkita tayo bukas']

# MediaPipe holistic setup
mp_holistic = mp.solutions.holistic

# Function to extract keypoints from holistic results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    return pose.flatten()

# Process video into a sequence of keypoints
def process_video(file_path):
    sequence = []
    cap = cv2.VideoCapture(file_path)
    step = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 30)

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        frame_count = 0
        while cap.isOpened() and len(sequence) < 30:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % step == 0:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
            frame_count += 1

    while len(sequence) < 30:
        sequence.append(np.zeros(99))

    return np.array([sequence])  # shape (1, 30, 99)

# API view to handle video prediction requests
class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        video = request.FILES.get('video')
        if not video:
            return Response({'error': 'No video uploaded'}, status=400)

        file_path = os.path.join(settings.MEDIA_ROOT, video.name)
        with open(file_path, 'wb+') as f:
            for chunk in video.chunks():
                f.write(chunk)

        sequence = process_video(file_path)
        prediction = model.predict(sequence)
        predicted_class = actions[np.argmax(prediction)]

        os.remove(file_path)  

        return Response({'prediction': predicted_class})
