import os
import numpy as np
import traceback  # Make sure this is imported at the top
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser

# Globals
actions = ['hello', 'ikinagagalak kong makilala ka', 'magkita tayo bukas']
model = None  # Lazy-loaded later


def download_model():
    import gdown
    model_path = settings.MODEL_PATH
    file_id = '1jF-UFZfHnpH-EASf3rgQ0x8XpLZayNkW'
    download_url = f'https://drive.google.com/uc?id={file_id}'

    if not os.path.exists(model_path):
        print("📥 Downloading model from Google Drive...")
        gdown.download(download_url, model_path, quiet=False)
    return model_path


def load_model():
    global model
    if model is None:
        import tensorflow as tf
        model_path = download_model()
        model = tf.keras.models.load_model(model_path)
    return model


def extract_keypoints(results):
    import numpy as np
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    return pose.flatten()


def process_video(file_path):
    import cv2
    import mediapipe as mp

    sequence = []
    cap = cv2.VideoCapture(file_path)
    step = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 30)

    with mp.solutions.holistic.Holistic(static_image_mode=False) as holistic:
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

    cap.release()

    while len(sequence) < 30:
        sequence.append(np.zeros(99))

    return np.array([sequence])  # shape (1, 30, 99)


class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        try:
            video = request.FILES.get('video')
            if not video:
                return Response({'error': 'No video uploaded'}, status=400)

            # Save file to MEDIA_ROOT
            file_path = os.path.join(settings.MEDIA_ROOT, video.name)
            with open(file_path, 'wb+') as f:
                for chunk in video.chunks():
                    f.write(chunk)

            print(f"📁 File saved to: {file_path}")

            # Process the video into keypoints
            sequence = process_video(file_path)
            print(f"📏 Sequence shape: {sequence.shape}")

            # Load and predict using the model
            model = load_model()
            print("🧠 Predicting...")
            prediction = model.predict(sequence)
            print(f"✅ Prediction result: {prediction}")

            predicted_class = actions[np.argmax(prediction)]
            print(f"🔤 Predicted class: {predicted_class}")

            # Clean up uploaded file
            os.remove(file_path)

            return Response({'prediction': predicted_class})
            
        except Exception as e:
            print("❌ Internal Server Error:")
            traceback.print_exc()  # <-- This prints full traceback to the logs
            return Response({'error': str(e)}, status=500)

