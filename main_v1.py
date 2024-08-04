import sys
sys.dont_write_bytecode = True

import os
import cv2
from model.siamese import SiameseModel

siamese_model = SiameseModel('model/siamesemodelv2.h5', 0.95, 0.75)

cap = cv2.VideoCapture(0)
while cap.isOpened():

    ret, frame = cap.read()

    frame = cv2.resize(frame, (1280, 720))
    #frame = frame[120:120+360, 440:440+360, :]

    frame = frame[120:120+480, 440:440+480, :]

    #Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
    # Save input image to application_data/input_image folder
        cv2.imwrite(os.path.join('data', 'input_data', 'input_img.jpg'), frame)
        # Run verification
        r,v = siamese_model.verify()
        print(r, v)


    cv2.imshow('verification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()