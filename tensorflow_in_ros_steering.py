"""
tensorflow_in_ros_mnist.py
Copyright 2016 Shunya Seiya

This software is released under the Apache License, Version 2.0
https://opensource.org/licenses/Apache-2.0
"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf

import argparse
import base64
import json
from PIL import Image as Image2
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import time
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

model = None

class RosTensorFlow():
    def __init__(self):
        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('steering_angle', Int16, queue_size=1)




    def callback(self, image_msg):

        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
        cv_image_binary = np.subtract(np.divide(np.array(cv_image_gray).astype(np.float32), 255.0), 0.5)
        #print(tf.shape(cv_image_binary))
        cv_image_car = cv2.resize(cv_image_binary,(160,320))
        image_array = np.reshape(cv_image_car,(160,320))


        transformed_image_array = image_array[None, :, :]
        #print(tf.shape(transformed_image_array))
        global graph
        with graph.as_default():
            steering_angle = float(model.predict(transformed_image_array, batch_size=1))

        print(steering_angle)
        #rospy.loginfo('%d' % steering_angle)
        self._pub.publish(steering_angle)



    def main(self):
        rospy.spin()

if __name__ == '__main__':

    with open("model.json", 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = "model.h5"  #args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    graph = tf.get_default_graph()

    rospy.init_node('rostensorflow')
    tensor = RosTensorFlow()
    tensor.main()


