import cv2
import pickle
import numpy as np
from torchvision import transforms
from keras.models import load_model,Model
from PIL import Image
import torch
import imutils
import os

model = load_model('inceptionv3.h5')

transform=transforms.Compose([transforms.Resize((299,299)),
							  transforms.ToTensor(),
							  transforms.Normalize(mean=[0.485,0.456,0.406],
							  std=[0.229,0.224,0.225])])



print("Loading Face Detector...")
protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
modelPath = os.path.sep.join(['face_detection_model', "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
le = pickle.loads(open('output/le.pickle', "rb").read())




def process_img(img_path,output_dir,filename):
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            t_img=transform(Image.fromarray(face))  #, 'RGB'
            t_img = torch.reshape(t_img, (1, 299, 299, 3))
            vec=model(t_img)

            i = recognizer.predict(vec)[0]
            name=le.classes_[i]
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
		    

            # draw the bounding box of the face along with the associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)

# Define test directory
test_dir = "test"
output_dir = "test_output"

os.makedirs(output_dir, exist_ok=True)

for person in os.listdir(test_dir):
    os.makedirs(f'{output_dir}/{person}', exist_ok=True)
    for filename in os.listdir(f'{test_dir}/{person}'):
        image_path = os.path.join(f'{test_dir}/{person}', filename)
        process_img(image_path,f'{output_dir}/{person}',filename)


print("Processed images saved in:", output_dir)