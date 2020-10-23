import numpy as np
import argparse
from cv2 import cv2

net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

image = cv2.imread("404902_2745393247130_736455659_n.jpg")
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()

cv2.imshow("Display window", image)
cv2.waitKey(0)