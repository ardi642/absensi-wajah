# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
# warnings.filterwarnings('ignore')

device_id = 0
model_dir = "./resources/anti_spoof_models"
model_test = AntiSpoofPredict(device_id)
image_cropper = CropImage()

def resize_image(image, width = 480, height = 640):
    height, width, channel = image.shape
    if width/height != 3/4:
        return cv2.resize(image, (width, height))
    else: 
        return image
    
# image dalam bentuk gbr
def is_spoofing(image):
    result = resize_image(image)
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))

    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        # start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        # test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:
        return {
            'type': 'real',
            'score': value
        }
    else:
        return {
            'type': 'fake',
            'score': value
        }
