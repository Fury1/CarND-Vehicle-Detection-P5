# Udacity Self Driving Car Nanodegree Project 5 Vehicle Detection
#
# Deep Learning Vehicle Detection
# Training, testing, and prediction pipeline to detect vehicles in video frames.

import functions
import cv2
import numpy as np
from keras.models import Sequential
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# # Uncomment this code to train the model if needed
# # Setup the model data to train and test on
# x_train, x_test, y_train, y_test = functions.model_data()
#
# # Setup the model for training
# model = functions.model(training=True)
# print(model.summary())
#
# functions.train_model(x_train, x_test, y_train, y_test,
#                       model,
#                       loss='mean_squared_error',
#                       batch_size=256,
#                       epochs=3)
#
# # Save the model weights when training is complete
# model.save_weights('model.h5')

# Load a test image from the test_images folder in RGB format
# image = cv2.cvtColor(cv2.imread('test_images/test4.jpg'), cv2.COLOR_BGR2RGB)

# Uncomment this code to make the video
# Set up the model for the video, load the trained weights
model = functions.model(input_shape=(720, 1280, 3)) # shape of video frame input
model.load_weights('model.h5')  # load the previous trained model

# Instantiate the heatmap class to keep track of moving averages to smooth frames
heat_average = functions.HeatAverage()

def process_image(image, model=model):

    # Create bounding box coordinates of the vehicles detected in the image/frame
    bboxes = functions.detect_vehicles(image, model)

    # Generate a heatmap from the bounding boxes found
    heatmap = functions.create_heatmap(image, bboxes)

    # Average a few frames of the heatmap data to improve stabilty, and threshold false postives
    heat_average.sanity_check(heatmap)

    # Draw a detection box around the vehicle based on the heatmap data
    result = functions.draw_boxes(image, heat_average.heatmap)

    return result

# Generate the completed project video
output = 'project_video_completed.mp4'
clip = VideoFileClip('project_video.mp4')
output_clip = clip.fl_image(process_image)
output_clip.write_videofile(output, audio=False)
