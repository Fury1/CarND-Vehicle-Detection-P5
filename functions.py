# Udacity Self Driving Car Nanodegree Project 5 Vehicle Detection
#
# Deep Learning Vehicle Detection Approach
# Helper functions for the vehicle detection pipeline.

import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.core import Lambda, Flatten, Dropout, Dense
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label


def model_data():
	"""
	Creates the training and testing data from the Udacity provided datasets.
	Randomly shuffles and plots the data for FCNN training and visualization.
	"""

	# Get all the file info for sample images of vehicles and non-vehicles
	vehicles = glob.glob('datasets\\vehicles\\*\\*.png')
	non_vehicles = glob.glob('datasets\\non-vehicles\\*\\*.png')

	# Read in image (RBG numpy array) and add corresponding labels to lists
	x_features = []
	y_labels = []

	# Vehicles are labeled as 1.0, non-vehicles are labeled as 0.0
	for image in vehicles:
		# Read the image, convert from BGR to RGB (cv2 scale is 0 - 255 for all image types)
		RGB_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
		x_features.append(RGB_image)
		y_labels.append(1.0)

	for image in non_vehicles:
		RGB_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
		x_features.append(RGB_image)
		y_labels.append(0.0)

	# Convert python list to numpy array for batch feeding Keras
	x_features = np.array(x_features)
	y_labels = np.array(y_labels)

	# Split the dataset into train and test sets, randomly shuffles data as well
	x_train, x_test, y_train, y_test = train_test_split(x_features, y_labels, test_size=0.3)

	# Explore the completed training dataset
	print('Number of training samples: {}'.format(len(x_train)))
	print('Training set image shape: {}'.format(x_train.shape))
	print('Number of vehicle training samples: {}'.format(np.sum(y_train == 1.0)))
	print('Number of non-vehicle training samples: {}'.format(np.sum(y_train == 0.0)))

	# Visualize some random images and their labels from the shuffled training set
	rows = 10
	cols = 5

	# Visualize the data
	fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
	fig.tight_layout()

	for row in range(rows):
		for col in range(cols):
			i = random.randint(0, len(x_train))
			axes[row, col].imshow(x_train[i], aspect='equal')
			axes[row, col].set_title(y_train[i])

	plt.show()

	return x_train, x_test, y_train, y_test


# Set up a model architecture to train for a classifier that will also
# arbitrarily expand its ouput on larger image sizes to detect vehicles
def model(input_shape=(64, 64, 3), training=False):
	"""
	FCNN: Fully Convolution Neural Network
	Classifier to train on vehicle and non-vehicle images.

	For training purposes the default input size must be used to generate a
	y_label for the error function (an ending 1x1x1 covolution output
	that gets flattened).

	When making predictions to detect vehicles, the input_shape will be larger
	and allow for vehicle predictions to be made in the frame. (convolutional output
	on the last layer will be expaneded arbitrarily based on the image/frame
	size passed in, last layer must not get flattened)

	input_shape: shape of the input image or video frame (int, tuple)
	"""

	model = Sequential()

	if training:
		# Center and normalize our data to 0 mean +-1
		model.add(Lambda(lambda x: (x - 128) / 128, input_shape=input_shape))

	# If not training, crop the input image to focus on the road only
	elif not training:

		# Hard code cropping values for the camera images on the car
		crop_top = 350
		crop_bottom = 150

		# Crop the input image to focus only on the road
		model.add(Cropping2D(cropping=((crop_top, crop_bottom), (0, 0)), input_shape=input_shape))
		# Center and normalize our data to 0 mean +-1
		model.add(Lambda(lambda x: (x - 128) / 128))

	model.add(Conv2D(80, (3, 3), activation='relu', padding="same"))
	model.add(Dropout(0.5))

	model.add(Conv2D(80, (3, 3), activation='relu', padding="same"))
	model.add(MaxPooling2D(pool_size=(10, 10)))

	model.add(Conv2D(80, (3, 3), activation='relu', padding="same"))

	model.add(Conv2D(1, (6, 6), activation="sigmoid")) # outputs a 1x1x1 covolution

	if training:
		model.add(Flatten()) # this allows the model to be trained with just convolutional layers

	return model


def train_model(x_train, x_test, y_train, y_test, model, loss, batch_size, epochs):
	"""
	Train the Keras model setup, plot the sessions for visualization.
	Takes a random test sample and makes a prediction once trained.

	x_train: training samples
	x_test: training labels
	y_train: test samples
	y_test: test labels
	model: Keras Sequential() model to train on (Keras model)
	loss: Keras model loss function to be used (string)
	batch_size: batch size for model training (int)
	epochs: number of training passes (int)
	"""

	# Setup an optimizer and loss function
	Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss=loss, optimizer=Adam)

	# Record the Keras history object for plotting and train the model
	keras_history_object = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

	# Plot the training and validation losses for visualization of this session
	plt.figure(1, figsize=(25, 15))
	plt.subplot(211)
	plt.plot(keras_history_object.history['loss'])
	plt.plot(keras_history_object.history['val_loss'])
	plt.title(loss)
	plt.ylabel(loss)
	plt.xlabel('Epoch')
	plt.legend(['Training set', 'Validation set'], loc='upper right')
	plt.show()

	# Make a prediction to check the trained classifier
	i = random.randint(0, len(x_test))
	plt.imshow(x_test[i])
	plt.title(y_test[i])
	reshaped_test = np.reshape(x_test[i], (1, 64, 64, 3))
	prediction = model.predict(reshaped_test)

	# Print the prediction to the prompt to verify against image visually being displayed
	print()
	print('Prediction: {}'.format(prediction))
	plt.show()


def detect_vehicles(image, model):
	"""
	Make a prediction for vehicles in the image.
	Creates bounding boxes for a detected vehicle using the
	trained model weights.

	image: image/frame for vehicle detection (numpy array)
	model: trained and loaded Keras model to make predictions with (Keras model with loaded weights)
	"""

	image_copy = np.copy(image)

	# Reshape image for Keras predictions, (samples, x, y, color_channels)
	hot_pixels = model.predict(np.reshape(image_copy, (1, image_copy.shape[0], image_copy.shape[1], image_copy.shape[2])))

	# Reshape the returned pixels to be plotted and visualized
	hot_pixels = np.reshape(hot_pixels, (hot_pixels.shape[1], hot_pixels.shape[2]))

	# Threshold the hot_pixels to get the highest predictions, discard lower values
	threshold = 0.85 # essentially confidence value of prediction
	hot_pixels[np.where(hot_pixels <= threshold)] = 0

	# Resize the returned pixels to match that of the orginal image crop before convolutions
	# 500 was the total height being cropped, see functions.model()
	hot_pixels = cv2.resize(hot_pixels, (image_copy.shape[1], image_copy.shape[0] - 500))

	# Average the data outward in x and y directions to make a consistant/smooth
	# hotspot from the thresholded pixels
	hot_pixels = gaussian_filter(hot_pixels, sigma=1)

	# Pad the cropped image to match its shape with the video frame
	# See functions.model() for cropped values (350, 150)
	hot_pixels = np.pad(hot_pixels, ((350, 150), (0, 0)), mode='constant', constant_values=0)

	# Create vehicle hot pixel (x, y) coordinates
	hot_pixel_coordinates = np.column_stack(hot_pixels.nonzero())

	# Setup a bounding box list and set a box size
	bboxes = []
	box_size = 90
	pixel_skip = 19 # number of samples to skip in hot_pixel_coordinates, speeds up compute time

	# Create the box coordinates to build a heatmap later ((x1, y1),(x2, y2)), center the boxes around the coordinates created
	for coordinate in hot_pixel_coordinates[::pixel_skip]:
		# Reverse coordinates for plotting from numpy array index
		bboxes.append(((coordinate[1] - (box_size // 2), coordinate[0] - (box_size // 2)), (coordinate[1] + (box_size // 2), coordinate[0] + (box_size // 2))))

	# Draw the boxes on the image copy to visualize the heatmap being built
	for bbox in bboxes:
		# Draw the box coordinates
		cv2.rectangle(image_copy,
					tuple(bbox[0]),
					tuple(bbox[1]),
					color=(0, 0, 255),
					thickness=3)

	# # Display for reference
	# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
	# fig.tight_layout()
	# ax1.imshow(image)
	# ax1.set_title('Original Image')
	# ax2.imshow(hot_pixels, cmap='hot')
	# ax2.set_title('Hot Pixels')
	# ax3.imshow(image_copy)
	# ax3.imshow(hot_pixels, cmap='hot', alpha=0.5)
	# ax3.set_title('Overlay')
	# ax4.imshow(image_copy, cmap='hot')
	# ax4.set_title('Bounding Boxes')
	# plt.show()

	return bboxes


def create_heatmap(image, bboxes):
	"""
	Create the heatmap of postive vehicle locations.

	image: original image or frame (numpy array)
	bboxes: vehicle detection bounding box coordinates ((x1, y1),(x2, y2))
	"""

	image_copy = np.copy(image)

	# Initialize heatmap array to all zeros in the shape of the original image
	heatmap = np.zeros((image.shape[0], image.shape[1]))

	# Make a heatmap by building heat inside the bounding box coordinates
	for box in bboxes:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# # Display for reference
	# plt.figure(figsize=(15, 15))
	# plt.imshow(image_copy)
	# plt.imshow(heatmap, cmap='hot', alpha=0.7)
	# plt.show()

	return heatmap


def draw_boxes(image, heatmap):
	"""
	Draw boxes around the detected vehicles and return a new modified image with
	vehicles outlined in boxes.

	image: original image or frame (numpy array)
	heatmap: a heat map of vehicles (numpy array)
	"""

	image = np.copy(image)

	labels = label(heatmap) # label the vehicle detections

	for car_number in range(1, labels[1] + 1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()

		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

		# Draw the box on the image
		cv2.rectangle(image, bbox[0], bbox[1], (0,0,255), 6)

	# # Display the final result
	# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 15))
	# fig.tight_layout()
	# ax1.imshow(image)
	# ax1.set_title('Cars Found: {}'.format(labels[1]))
	# ax2.imshow(labels[0], cmap='gray')
	# ax2.set_title('Final Result')
	# plt.show()

	return image


class HeatAverage():
	"""
	Heatmap class that is used to keep track of a few heatmaps and average
	them together to smooth vehicle detection. Also thresholds values to remove false postives.
	"""

	def __init__(self):

		# Number of past heat maps to keep for moving averages
		self.frame_count = 12

		# List of past frame heatmaps
		self.heatmaps = []

		# Initialize with an empty heatmap
		self.heatmap = np.zeros((720, 1280)) # shape of heatmap input

	def sanity_check(self, heatmap):

		# False positive threshold
		threshold = 135

		# Gather some heatmap data to save if there isn't enough
		if len(self.heatmaps) < self.frame_count:

			self.heatmap = heatmap

			# Threshold the heatmap to remove false postives
			self.heatmap[self.heatmap <= threshold] = 0

			# Add it to the list to collect some frames
			self.heatmaps.append(heatmap)

		else:
			# Remove the oldest heatmap to make room for a new one
			del self.heatmaps[0]

			# Add the new heatmap to the list of heatmaps
			self.heatmaps.append(heatmap)

			# Average the heatmaps available in the list
			self.heatmap = np.average(self.heatmaps, axis=0)

			# Threshold the averaged heatmaps to remove false postives
			self.heatmap[self.heatmap <= threshold] = 0
