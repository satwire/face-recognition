import os
import numpy as np
import face_recognition
import pickle
from cv2 import cv2
from imutils import paths

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(tuple((img, os.path.splitext(filename)[0])))
    return images

def face_recognition_from_image(image, image_number, net):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            if startX <= w or startY <= h:
                cv2.imwrite(os.path.join(faces_path , 'image' + str(image_number) + '_face' + str(i) + '.jpg'), image[startY:endY, startX:endX])
                # text = "{:.2f}%".format(confidence * 100)
                # y = startY - 10 if startY - 10 > 10 else startY + 10
                # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                # cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    # cv2.imshow("Display window", image)
    # cv2.waitKey(0)

net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

dataset_path = os.path.realpath(os.getcwd() + '/dataset')
faces_path = os.path.realpath(os.getcwd() + '/faces')
images = load_images_from_folder(dataset_path)

for i in range(len(images)):
    face_recognition_from_image(images[i][0], i, net)

image_paths = list(paths.list_images(faces_path))
data = []

for i in range(len(image_paths)):
    print("Processing image {}/{}".format(i + 1, len(image_paths)))
    print(image_paths[i])
    image = cv2.imread(image_paths[i])
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes = face_recognition.face_locations(rgb, 1, 'hog')
    
    encodings = face_recognition.face_encodings(rgb, boxes)

    d = [{"image_path": image_paths[i], "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
    data.extend(d)

print("Serializing encodings")
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()