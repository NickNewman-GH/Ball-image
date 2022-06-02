import cv2
from tensorflow.keras.models import (load_model)
import numpy as np

classification_model = load_model('ball_classification_model.h5')
roi_model = load_model('ball_roi_model.h5')

cap = cv2.VideoCapture(0)
image_shape = (160, 120, 3)
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = frame[:,(640 - 360)//2:640 - (640 - 360)//2]
    frame = cv2.resize(frame, (image_shape[1], image_shape[0]))
    frame = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    predicted_class = classification_model.predict(np.array([frame]))
    print(predicted_class[0], end = '\r')
    if predicted_class[0] > 0.7:
        roi = roi_model.predict(np.array([frame]))[0]
        roi = list(map(int,[roi[0]*image_shape[1],roi[1]*image_shape[0],roi[2]*image_shape[1],roi[3]*image_shape[0]]))
        cv2.rectangle(frame,(roi[0],roi[1]),(roi[2],roi[3]),(255,0,0), 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (image_shape[1]*3, image_shape[0]*3))

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break