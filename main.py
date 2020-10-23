import numpy as np
import argparse
from cv2 import cv2

net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

image = cv2.imread("404902_2745393247130_736455659_n.jpg")
cv2.imshow("Display window", image)
cv2.waitKey(0)