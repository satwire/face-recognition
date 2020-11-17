import os
import numpy as np
import face_recognition
import pickle
import dlib
from cv2 import cv2
from sklearn.cluster import DBSCAN
from imutils import build_montages
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

            distanceX = endX - startX
            distanceY = endY - startY
            
            startX -= (int)(distanceX * 0.25)
            endX += (int)(distanceX * 0.25)
            startY -= (int)(distanceY * 0.25)
            endY += (int)(distanceY * 0.25)
            
            if startX < 0: startX = 0
            if endX > w: endX = w
            if startY < 0: startY = 0
            if endY > h: endY = h

            if startX <= w or startY <= h:
                cv2.imwrite(os.path.join(faces_path , 'image' + str(image_number) + '_face' + str(i) + '.jpg'), image[startY:endY, startX:endX])
                # text = "{:.2f}%".format(confidence * 100)
                # y = startY - 10 if startY - 10 > 10 else startY + 10
                # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                # cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    # cv2.imshow("Display window", image)
    # cv2.waitKey(0)

def face_processing_from_images(image_paths):
    data = []   
    for i in range(len(image_paths)):
        print("Processing image {}/{}".format(i + 1, len(image_paths)))
        print(image_paths[i])
        image = cv2.imread(image_paths[i])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = face_recognition.face_locations(rgb, 1, "cnn")
        
        encodings = face_recognition.face_encodings(rgb, boxes)

        d = [{"image_path": image_paths[i], "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
        data.extend(d)    
    return data

net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# set directories
dataset_path = os.path.realpath(os.getcwd() + '/dataset')
faces_path = os.path.realpath(os.getcwd() + '/faces')
images = load_images_from_folder(dataset_path)

# extract faces from dataset 
for i in range(len(images)):
    face_recognition_from_image(images[i][0], i, net)

# process faces
image_paths = list(paths.list_images(faces_path))
data = face_processing_from_images(image_paths)

# write serialized faces
print("Serializing encodings")
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()

# load serialized faces
data = pickle.loads(open("encodings.pickle", "rb").read())
data = np.array(data)
encodings = [d["encoding"] for d in data]

clt = DBSCAN(metric="euclidean", n_jobs=-1)
clt.fit(encodings)

label_ids = np.unique(clt.labels_)
unique_faces_count = len(np.where(label_ids > -1)[0])
print("{} unique faces.".format(unique_faces_count))

for label_id in label_ids:
    print("[INFO] faces for face ID: {}".format(label_id))
    idxs = np.where(clt.labels_ == label_id)[0]
    idxs = np.random.choice(idxs, size=min(25, len(idxs)),
        replace=False)

    faces = []
    for i in idxs:
        image = cv2.imread(data[i]["image_path"])
        (top, right, bottom, left) = data[i]["loc"]
        face = image[top:bottom, left:right]

        face = cv2.resize(face, (96, 96))
        faces.append(face)

    montage = build_montages(faces, (96, 96), (5, 5))[0]
    title = "Face ID #{}".format(label_id)
    title = "Unknown Faces" if label_id == -1 else title
    cv2.imshow(title, montage)
    cv2.waitKey(0)
