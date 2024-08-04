from keras.models import Model, load_model
from keras.losses import BinaryCrossentropy
from model.distanceLayer import DistanceLayer
from threading import Thread
from preprocess.preprocess import preprocess_image

import numpy as np

import os

class SiameseModel():

    def __init__(self, pre_model_path, detection_threshold = 0.5, verification_threshold = 0.5):
        
        self.model = load_model(
            pre_model_path,
            custom_objects = {'DistanceLayer': DistanceLayer,
                                'BinaryCrossentropy': BinaryCrossentropy
                            }
        )

        self.detection_threshold = detection_threshold
        self.verification_threshold = verification_threshold

    def verify(self):
        results = []
        for image in os.listdir(os.path.join('data','val_data')):
            
            input_img = preprocess_image(os.path.join('data', 'input_data', 'input_img.jpg'))
            val_img = preprocess_image(os.path.join('data', 'val_data', image))
            
            #make predict
            result = self.model.predict(list(np.expand_dims([input_img, val_img], axis=1)))
            results.append(result)

        #print(results)

        print(len(results))

        # Detection Threshold: Metric above which a prediciton is considered positive
        detection = np.sum(np.array(results) > self.detection_threshold)

        print(detection)

        # Verification Threshold: Proportion of positive predictions / total positive samples
        verification = detection / len(os.listdir(os.path.join('data', 'val_data')))
        
        print(verification)

        #print(self.verification_threshold)

        verified = verification > self.verification_threshold

        print(verified)

        return results, verified
