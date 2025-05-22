import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('emotion_class_tf.keras')

emotion_labels = ['Anger', 'Contempt', 'Disgust',
                  'Happy', 'Fear', 'Neutral', 'Sad', 'Surprise']

# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # To open the webcam

IMG_SIZE = 48  # x48

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop and preprocess face
        roi_gray = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.expand_dims(
            roi_normalized, axis=(0, -1))  # Shape: (1, 48, 48, 1)

        # Predict emotion
        prediction = model.predict(roi_reshaped, verbose=0)
        emotion_idx = int(np.argmax(prediction))
        emotion_text = emotion_labels[emotion_idx]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('CNN Emotion Recognition | Press q to exit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
