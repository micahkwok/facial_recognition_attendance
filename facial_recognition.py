import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import face_recognition
import cv2
from datetime import datetime
import os

# Supresses tensorflow warning msg in terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Loading...")

# Creating empty csv - consider parameterizing this later
today = datetime.today()

empty_frame = pd.DataFrame({'Name': [], 'Time': []})
empty_frame.to_csv(f'{today.strftime("%b-%d-%Y")}_attendance_register.csv', index = False)

def mark_attendance(name):
    with open(f'{today.strftime("%b-%d-%Y")}_attendance_register.csv','r+') as f:
        DataList = f.readlines()
        names = []
        for data in DataList:
            ent = data.split(',')
            names.append(ent[0])
        if name not in names:
            curr = datetime.now()
            dt = curr.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt}')
            print(f'{name} marked in attendance book')

# Setting up Labels
labels = pd.read_csv('converted_keras/labels.txt', header = None)

labels = labels.rename(columns = {0:'Names'})
new_index = []
new_labels = []

for x in labels['Names']:
    new_index.append(int(x.split(" ", 1)[0]))  # to ensure it is not appended as a list
    new_labels.append(x.split(" ", 1)[1])
    
prediction_dict = dict(zip(new_index, new_labels))

# Setting up Model
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model (set compile to False)
model = tensorflow.keras.models.load_model('converted_keras/keras_model.h5', compile = False)

# Computer Vision
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True:
    success, img = cap.read()
    res = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    normalized_image_array = (res.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)

    predicted_score = round(prediction.max(),2)
    predicted_name = prediction_dict[prediction.argmax()]

    # print(prediction)
    # print(prediction_dict[prediction.argmax()])

    # Face Location
    face_locations = face_recognition.face_locations(img)

    # Only when face_recogniiton sees a face is the prediction even possibly marked, this way to stop triggering mark attendance when no faces are present
    if len(face_locations) != 0:

        for face_location in face_locations:
            # Print the location of each face in this image
            top, right, bottom, left = face_location

            # We can draw rectangle using OpenCV rectangle method
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)   # Face Rectangle
            cv2.rectangle(img, (left, bottom-35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,predicted_name, (left+6, bottom-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)     # Name Text
            cv2.putText(img,str(predicted_score) + " probability", (left+6, bottom+70), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)  # Probability
            
        # Only mark attendance if predicted score is greater than 0.90    
        if predicted_score > 0.90:
            mark_attendance(predicted_name)

    cv2.imshow("Webcam",img)
    cv2.waitKey(1)

    # Pressing 'q' stops the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break

print("Shutting down...")

cap.release()
