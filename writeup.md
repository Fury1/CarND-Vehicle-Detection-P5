# **Vehicle Detection Project**

The goals of this project are the following:

* Train a classifier to correctly identify vehicles
* Develop a software pipeline to detect nearby vehicles on the road from a camera on the front of a moving vehicle


[//]: # (Image References)
[image1]: ./output_images/visualized_training_data.png
[image2]: ./output_images/prediction.png
[image3]: ./output_images/training_plot.png
[image4]: ./output_images/hot_pixels.png
[image5]: ./output_images/bounding_boxes_complete.png
[image6]: ./output_images/different_scale.png
[image7]: ./output_images/different_scale2.png
[image8]: ./output_images/heatmap.png
[image9]: ./output_images/box_labels.png
[image10]: ./output_images/detected_vehicles.png

---

### Approach

During the initial implementation of this project I had studied the historical approach of using an SVM classifier against HOG features across a sliding window search. After doing some reading, it was brought to my attention that using a **deep learning approach** could be much more effective and computationally efficient to detect vehicles in a camera frame. I decided to try and tackle this project using deep learning and a Fully Convolutional Neural Network (FCNN) along with a few traditional computer vision techniques. After watching [this CS231n](https://www.youtube.com/watch?v=_GfPYLNQank) lecture I was confident that I could make a FCNN work for the project pipeline.

### Sample Features

To begin the project I began by creating and exploring my dataset. Udacity provided a great vehicle/non-vehicle dataset from the GTI vehicle image database, the KITTI vision benchmark suite, as well as a few examples manually extracted from the project video.

The dataset had an even balance of vehicles/non-vehicles with a good variety of examples. This was determined by simply looking at the number of samples in the vehicles vs non-vehicles training samples (lines 56-59 `functions.py`).

```
Number of training samples: 13154
Training set image shape: (13154, 64, 64, 3)
Number of vehicle training samples: 6529
Number of non-vehicle training samples: 6625
```

Once the balance of the data was verified, it was randomly shuffled and split for training/testing (70-30 split, line 53 `functions.py` ) and visualized afterward. (0.0 indicates a non-vehicle label, 1.0 is a vehicle)

 ![Training Data Examples][image1]

Ultimately the trained classifier is run on a video stream to find vehicles in each frame. This makes the video stream the final test on unseen data.

---

### FCNN Model Architecture, Parameters, and Training

A FCNN was used as a deep learning approach to classify vehicles. My model architecture started as just a few convolutional layers with dropout to prevent over fitting. From there I trained and tested different configurations to see what worked best, ultimately I found that the below architecture gave me the best results and trained easily. The only prerequisite my network had to have was an ending convolutional layer with an output of 1x1x1 for a 64x64 input image (training image size). I will expand upon this more below.

My final FCNN architecture is as follows (lines 80-130 `functions.py`):

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 64, 64, 3)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 80)        2240
_________________________________________________________________
dropout_1 (Dropout)          (None, 64, 64, 80)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 64, 80)        57680
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 80)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 6, 80)          57680
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 1, 1)           2881
_________________________________________________________________
flatten_1 (Flatten)          (None, 1)                 0
=================================================================
Total params: 120,481
Trainable params: 120,481
Non-trainable params: 0
_________________________________________________________________

```

The features were first normalized to a 0 mean +-1, cropped to a specified "search window", then fed into a series of convolutions that ultimately end in a 1x1x1 convolution (64x64 input only). The ending 1x1x1 output during training is an import step,  its important because that ending layer can be flattened to just 1, which allows for a label value to be produced. You could set this network up with a tanh activation function on the last layer to output a 0 or 1, however, I decided to use a sigmoid activation on my flattened output layer. I liked the idea of using the produced float as more of a probabilistic method of determining whether or not there was a vehicle.

During training an adam optimizer and mean squared error loss was used to train the network.
I trained the FCNN on lines 21-25 `p5.py` in batch sizes of 256 for 3 epochs. A random prediction was also made for a sample at the end of training to be compared. (lines 133-176 `functions.py`)

```

13154/13154 [==============================] - 10s - loss: 0.1382 - val_loss: 0.0750
Epoch 2/3
13154/13154 [==============================] - 7s - loss: 0.0441 - val_loss: 0.0381
Epoch 3/3
13154/13154 [==============================] - 7s - loss: 0.0217 - val_loss: 0.0261

[[ 0.99902594]] <--- Prediction on random sample: (vehicle, 1.0)

```
![Prediction Sample][image2]
![Training][image3]

---

### Searching For Vehicles

The great part about a trained FCNN is that even though we trained it on 64x64 images to detect cars. We can re-use the same network (minus the last flattened layer we needed for training) to search for cars everywhere in an image that is larger then 64x64. Because there are no fully connected layers, just convolutional layers, the output can arbitrarily scale to whatever the input is. Instead of having a 1x1x1 output, larger images produce a map of predictions, which can be used as an alternative to the traditional ML sliding window search. Below is the raw output of an image larger then 64x64 with two cars in it. The output has been plotted with highest values as hot spots or in our case identified vehicles.

![Raw Prediction Map][image4]

This prediction map can be used to draw bounding boxes that then create the final heat map of vehicles. This is also a great place to start thresholding values and clean up noise in the predictions. Below is an example of the predictions that have been resized, averaged, and thresholded to smooth everything out. I found from experimenting a bit that this gave me a better end result for my bounding boxes. (lines 179-249 `functions.py`)

![Detection Pipeline][image5]

I think it is also worth noting that because we are using a deep learning network, the scale of the vehicle in the frame does not matter when making predictions so long as the network is trained properly.

![Detection Pipeline][image6]
![Detection Pipeline][image7]

Once the bounding boxes have been determined a heat map is built. (lines 252-277 `functions.py`)

The heat map is made by adding +1 to all pixel values in each bounding box. Where the boxes overlap more heat is generated (combines/compounds overlaps), which yields an accurate location of the vehicle in the frame while being able to better threshold "lower heat" which would indicate false positives (less box overlaps = outliers/false positives).

For my implementation I also decided to make a heat map class that is used to average a few frames of heat maps that are then thresholded to remove false positives in lines 319-363 of `functions.py` and line 50 `p5.py`.
Again, from experimenting I decided that this approach gave me the most smooth consistent heat map when processing the project video.

Final averaged and thresholded heat map:

![Heatmap][image8]

After the final heat map is produced it is fed into a draw box function which creates labels from the heat map using the "label" function from the "scipy.ndimage.measurements" library (line 291 `functions.py`).  It essentially creates islands for all values that are grouped together that do no equal zero.

Output:

![Vehicle Labels][image9]

From the identified "islands" we can draw boxes around them and return the final image with the vehicles detected accurately (lines 280-316 `functions.py`).


![Detected Vehicles][image10]

My final choice of parameters (thresholds, frame averages, etc) were determined through trial and error. I spent a ton of time changing parameters to get a feel for how they affected features.
I tuned and checked until I had a good reliable result across multiple test images. From there I processed the video, re-tuned and re-checked problem areas to correct any issues I saw.
Eventually I settled on parameters that worked best for my classifier, computer vision techniques implemented, and problem areas.

---

### Video Implementation
Here's a [link to my video result](./project_video_completed.mp4)

---

## Discussion

### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* One of the main issues that I faced during my implementation of this project was removing the false positives as well as not detecting cars on the other side of the road inadvertently. It took me a while to figure out the right combination of parameters that would remove the false positives and limit detections on the other side of the road while also not ruining vehicle detection on my side of the road.
I theorized as best I could, but also had to do a lot of experimenting to get the right setup that produced the desired result.

* My pipeline is most likely to fail on a road with more cars or different types of landscapes, weather, road conditions, etc. The dataset that was provided from Udacity was great for this project, however, I can say with confidence that the dataset does not have enough samples to cover all real world scenarios. I can also say that my network is definitely not large enough to generalize a larger dataset with more real world examples.

* To make my implementation more robust I think that my model architecture could be increased in size to include more parameters that could generalize a larger dataset that better reflects a larger spread of vehicle/non-vehicle examples. I also think image color space manipulation could be beneficial in extracting vehicles before they are feed into the network as a means of preprocessing information, that could possible help combat changing road conditions, weather, and image noise.
