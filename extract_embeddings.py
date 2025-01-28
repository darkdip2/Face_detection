# import libraries
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import torch.nn as  nn
import tensorflow as tf
from torchvision import transforms
from keras.models import load_model,Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications import VGG16
from tensorflow.keras.layers import Input
import torchvision.models as models
from PIL import Image
import torch


'''model = models.resnet50(pretrained=True)  #Using Resnet feature extraction
layer = model._modules.get('avgpool')

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_vector(image):
    t_img = transform(image)
    my_embedding = torch.zeros(2048)
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())                 # <-- flatten

    h = layer.register_forward_hook(copy_data)
    with torch.no_grad():                               # <-- no_grad context
        model(t_img.unsqueeze(0))                       # <-- unsqueeze
    h.remove()
    return my_embedding'''



'''
#base_model = InceptionV3(weights='imagenet')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
#model=VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(299,299,3)))
#model.save('inceptionv3.h5')
#model.save('vgg16.h5')'''

#model = load_model('inceptionv3.h5')
model=load_model('vgg16.h5')
#print(model.summary())



transform=transforms.Compose([transforms.Resize((299,299)),
							  transforms.ToTensor(),
							  transforms.Normalize(mean=[0.485,0.456,0.406],
							  std=[0.229,0.224,0.225])])







# load serialized face detector
print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load serialized face embedding model
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# grab the paths to the input images in our dataset
print("Quantifying Faces...")
imagePaths = list(paths.list_images("dataset"))

# initialize our lists of extracted facial embeddings and corresponding people names
knownEmbeddings = []
knownNames = []

# initialize the total number of faces processed
total = 0

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	if (i%50 == 0):
		print("Processing image {}/{}".format(i, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also means our minimum probability test (thus helping filter out weak detections)
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			#faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				#(96, 96), (0, 0, 0), swapRB=True, crop=False)
			#embedder.setInput(faceBlob)
			#vec = embedder.forward()
			t_img=transform(Image.fromarray(face))  #, 'RGB'
			t_img = torch.reshape(t_img, (1, 299, 299, 3))
			#print(t_img.shape)
			#t_img=tf.keras.applications.inception_v3.preprocess_input(t_img)
			vec=model(t_img) 
			#print(vec.shape)

		
			# add the name of the person + corresponding face embedding to their respective lists
			knownNames.append(name)
			knownEmbeddings.append(vec)
			total += 1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()