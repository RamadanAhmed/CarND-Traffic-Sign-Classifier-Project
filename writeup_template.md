# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plots/training_data_dist.png "Visualization"
[image2]: ./plots/preprocess-img.png "pre-process result"
[image3]: ./plots/augmentation.png "Random Noise"
[image4]: ./plots/test-data.png "Traffic signs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the information on other color channel will not make difference and decrease dimensonality of the input will make network converge easily

As a last step, I normalized the image data because to have zero mean and variance will have better convergence in optimization

And Here images of result preprocessing

![alt text][image2]

I decided to generate additional data because dataset is imbalanced so I have to add more examples for minority samples so network can't overfit the training set 

To add more data to the the data set, I used `tensorflow.keras.preprocessing.image.ImageDataGenerator` to generate image for minority samples by applying shear, rotation, height shifting, width shifting and add some gaussian noise to image

Here is an example of an original image and an augmented image:

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:
___________________________________________________________________________
Layer (type)            |     Output Shape          |     Param            |
|:---------------------:|:-------------------------:|:--------------------:| 
conv2d_6 (Conv2D)       |  (None, 28, 28, 6)        |    156               |
dropout_12 (Dropout)    |  (None, 28, 28, 6)        |     0                |
average_pooling2d_6     |   (None, 14, 14, 6)       |     0                |
conv2d_7 (Conv2D)       |   (None, 10, 10, 16)      |    2416              | 
dropout_13 (Dropout)    |   (None, 10, 10, 16)      |     0                |
average_pooling2d_7     |   (None, 5, 5, 16)        |     0                |
flatten_3 (Flatten)     |   (None, 400)             |     0                |
dense_9 (Dense)         |   (None, 120)             |    48120             |
dropout_14 (Dropout)    |   (None, 120)             |     0                | 
dense_10 (Dense)        |   (None, 84)              |    10164             |
dropout_15 (Dropout)    |   (None, 84)              |     0                |
dense_11 (Dense)        |   (None, 43)              |    3655              |

Total params: 64,511
Trainable params: 64,511
Non-trainable params: 0
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an `Adam` optimizer with learning rate of `0.001` and batch size of `500` and `100` epoch

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 90.13 %
* validation set accuracy of 94.81 % 
* test set accuracy of 93.28%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are German traffic signs that I found on the web:

![alt text][image4]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
Speed limit (20km/h)	                         | 	Vehicles over 3.5 metric tons prohibited     |
Speed limit (30km/h)	                         | 	Speed limit (20km/h)                         |
No passing for vehicles over 3.5 metric tons	 | 	No passing for vehicles over 3.5 metric tons |
Right-of-way at the next intersection	         | 	Right-of-way at the next intersection        |
Priority road	                                 | 	Priority road                                |
Yield	                                         | 	Yield                                        |
Stop	                                         | 	Stop                                         |
No vehicles	                                     | 	No vehicles                                  |
Vehicles over 3.5 metric tons prohibited	     | 	Vehicles over 3.5 metric tons prohibited     |
No entry	                                     | 	No entry                                     |
General caution	                                 | 	General caution                              |
Dangerous curve to the left	                     | 	Dangerous curve to the left                  |
Speed limit (50km/h)	                         | 	Speed limit (50km/h)                         |
Dangerous curve to the right	                 | 	Dangerous curve to the left                  |
Double curve	                                 | 	Double curve                                 |
Bumpy road	                                     | 	Bumpy road                                   |
Slippery road	                                 | 	Slippery road                                |
Road narrows on the right	                     | 	Road narrows on the right                    |
Road work	                                     | 	Road work                                    |
Traffic signals	                                 | 	Traffic signals                              |
Pedestrians	                                     | 	Pedestrians                                  |
Children crossing	                             | 	Right-of-way at the next intersection        |
Bicycles crossing	                             | 	Slippery road                                |
Speed limit (60km/h)	                         | 	Speed limit (20km/h)                         |
Beware of ice/snow	                             | 	Right-of-way at the next intersection        |
Wild animals crossing	                         | 	Wild animals crossing                        |
End of all speed and passing limits	             | 	End of all speed and passing limits          |
Turn right ahead	                             | 	Turn right ahead                             |
Turn left ahead	                                 | 	Turn left ahead                              |
Ahead only	                                     | 	Ahead only                                   |
Go straight or right	                         | 	Go straight or right                         |
Go straight or left	                             | 	Go straight or left                          |
Keep right	                                     | 	Keep right                                   |
Keep left	                                     | 	Keep left                                    |
Speed limit (70km/h)	                         | 	Speed limit (70km/h)                         |
Roundabout mandatory	                         | 	Priority road                                |
End of no passing	                             | 	End of no passing                            |
End of no passing by vehicles over 3.5 metric tons| 	End of no passing                        |
Speed limit (80km/h)	                         | 	Speed limit (20km/h)                         |
End of speed limit (80km/h)	                     | 	Children crossing                            |
Speed limit (100km/h)	                         | 	Speed limit (100km/h)                        |
Speed limit (120km/h)	                         | 	Speed limit (20km/h)                         |
No passing	                                     | 	End of no passing                            |


The model was able to correctly guess 30 of the 43 traffic signs, which gives an accuracy of 69.77%. which is less than the accuracy on test-set

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


| Sign         	|     Probabilites	        					|     Probabilites	        |  
|:---------------------:|:---------------------------------------------:|--------------------------:| 
| Speed limit (20km/h) | [0.82269317 0.11334925 0.03322334 0.02258257 0.00307689] | ['Vehicles over 3.5 metric tons prohibited', 'Speed limit (20km/h)', 'Dangerous curve to the left', 'No passing', 'Go straight or left'] |
| Speed limit (30km/h) | [6.5254879e-01 3.2186776e-01 2.3470091e-02 1.9146878e-03 1.6326459e-04] | ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (70km/h)', 'Speed limit (50km/h)', 'Speed limit (120km/h)'] |
| Speed limit (50km/h) | [9.5590889e-01 3.8057514e-02 5.7986956e-03 2.1709075e-04 8.7021090e-06] | ['No passing for vehicles over 3.5 metric tons', 'No passing', 'End of no passing by vehicles over 3.5 metric tons', 'Slippery road', 'End of no passing'] |
| Speed limit (60km/h) | [9.6911901e-01 3.0875618e-02 5.3050567e-06 2.9166637e-08 9.8330499e-09] | ['Right-of-way at the next intersection', 'Beware of ice/snow', 'Double curve', 'Roundabout mandatory', 'Children crossing'] |
| Speed limit (70km/h) | [0.53843474 0.16894907 0.11759134 0.09189451 0.07055765] | ['Priority road', 'Speed limit (60km/h)', 'Speed limit (100km/h)', 'Stop', 'Speed limit (80km/h)'] |
| Speed limit (80km/h) | [1.0000000e+00 5.8237393e-10 2.1693251e-11 1.1710580e-11 2.4695434e-12] | ['Yield', 'No vehicles', 'No passing', 'End of all speed and passing limits', 'Stop'] |
| End of speed limit (80km/h) | [9.9395508e-01 2.7785278e-03 1.8973427e-03 5.2238768e-04 2.4102634e-04] | ['Stop', 'Roundabout mandatory', 'Speed limit (100km/h)', 'Speed limit (20km/h)', 'Keep left'] |
| Speed limit (100km/h) | [9.9981385e-01 1.7403302e-04 4.2205838e-06 3.1983450e-06 1.9096478e-06] | ['No vehicles', 'Yield', 'Stop', 'Speed limit (60km/h)', 'Priority road'] |
| Speed limit (120km/h) | [9.9999726e-01 1.6148210e-06 1.0518258e-06 9.1070424e-12 5.1819742e-12] | ['Vehicles over 3.5 metric tons prohibited', 'No passing', 'End of no passing', 'Keep left', 'End of all speed and passing limits'] |
| No passing | [9.9999893e-01 6.9964392e-07 2.4778583e-07 8.0576108e-08 6.6289800e-09] | ['No entry', 'Stop', 'No passing', 'Speed limit (100km/h)', 'No passing for vehicles over 3.5 metric tons'] |
| No passing for vehicles over 3.5 metric tons | [9.9919301e-01 8.0695667e-04 1.2286531e-08 1.5620198e-11 1.8317390e-12] | ['General caution', 'Traffic signals', 'Pedestrians', 'Road narrows on the right', 'Bumpy road'] |
| Right-of-way at the next intersection | [9.9999905e-01 9.5088348e-07 4.6312735e-08 4.3041806e-10 1.2066878e-10] | ['Dangerous curve to the left', 'Double curve', 'Slippery road', 'Road narrows on the right', 'Bumpy road'] |
| Priority road | [9.8568475e-01 1.2518990e-02 1.2122713e-03 4.0913315e-04 1.2238158e-04] | ['Speed limit (50km/h)', 'Speed limit (20km/h)', 'Speed limit (80km/h)', 'Speed limit (30km/h)', 'Speed limit (60km/h)'] |
| Yield | [0.43761584 0.26519707 0.13438423 0.10924663 0.02008642] | ['Dangerous curve to the left', 'Dangerous curve to the right', 'Slippery road', 'Go straight or right', 'Children crossing'] |
| Stop | [9.6204573e-01 2.7261274e-02 9.7998986e-03 8.6683995e-04 2.1932923e-05] | ['Double curve', 'Dangerous curve to the left', 'Wild animals crossing', 'Slippery road', 'Road work'] |
| No vehicles | [9.9997342e-01 1.8549654e-05 6.4723508e-06 6.3805101e-07 6.0186051e-07] | ['Bumpy road', 'Bicycles crossing', 'Children crossing', 'Road work', 'Slippery road'] |
| Vehicles over 3.5 metric tons prohibited | [0.6785345  0.2675281  0.04671803 0.00597898 0.00104046] | ['Slippery road', 'Dangerous curve to the left', 'Road work', 'Double curve', 'Bumpy road'] |
| No entry | [9.8866278e-01 1.1255221e-02 8.0478800e-05 1.4782369e-06 1.9851106e-08] | ['Road narrows on the right', 'Pedestrians', 'General caution', 'Traffic signals', 'Wild animals crossing'] |
| General caution | [9.9905807e-01 5.0293218e-04 2.1485396e-04 7.0657588e-05 7.0485294e-05] | ['Road work', 'Road narrows on the right', 'Slippery road', 'Wild animals crossing', 'Double curve'] |
| Dangerous curve to the left | [9.9996448e-01 3.5485824e-05 1.1677099e-09 1.7262694e-10 1.0099621e-12] | ['Traffic signals', 'General caution', 'Road work', 'Road narrows on the right', 'Pedestrians'] |
| Dangerous curve to the right | [0.4509171  0.29314214 0.1843623  0.0673847  0.00418373] | ['Pedestrians', 'Road narrows on the right', 'Traffic signals', 'General caution', 'Road work'] |
| Double curve | [0.82644945 0.07947829 0.0523022  0.03788973 0.00323828] | ['Right-of-way at the next intersection', 'Beware of ice/snow', 'Children crossing', 'Double curve', 'Road work'] |
| Bumpy road | [0.9420989  0.03173924 0.00845848 0.00693439 0.00632601] | ['Slippery road', 'Children crossing', 'Dangerous curve to the left', 'Double curve', 'Wild animals crossing'] |
| Slippery road | [5.7030892e-01 2.9959652e-01 1.2894876e-01 1.1189189e-03 2.4997440e-05] | ['Speed limit (20km/h)', 'Speed limit (60km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (80km/h)'] |
| Road narrows on the right | [7.1549225e-01 1.9582027e-01 8.1076354e-02 5.3771185e-03 5.4756348e-04] | ['Right-of-way at the next intersection', 'Double curve', 'Beware of ice/snow', 'Road work', 'Slippery road'] |
| Road work | [5.7727277e-01 3.6993122e-01 5.1389478e-02 1.3517655e-03 2.9693752e-05] | ['Wild animals crossing', 'Slippery road', 'Dangerous curve to the left', 'Double curve', 'Road work'] |
| Traffic signals | [6.0704809e-01 3.6716166e-01 2.5609525e-02 1.8044085e-04 1.9060977e-07] | ['End of all speed and passing limits', 'End of no passing', 'End of speed limit (80km/h)', 'End of no passing by vehicles over 3.5 metric tons', 'Vehicles over 3.5 metric tons prohibited'] |
| Pedestrians | [1.0000000e+00 4.8991837e-09 1.9718147e-09 1.4862507e-09 1.2318296e-10] | ['Turn right ahead', 'Keep left', 'Ahead only', 'Stop', 'Go straight or right'] |
| Children crossing | [9.9679607e-01 2.9996394e-03 5.9114893e-05 5.2157768e-05 2.3898063e-05] | ['Turn left ahead', 'Ahead only', 'Roundabout mandatory', 'Speed limit (80km/h)', 'Keep right'] |
| Bicycles crossing | [9.9999714e-01 2.8571280e-06 1.2213650e-08 2.2200026e-09 4.3536214e-10] | ['Ahead only', 'Go straight or right', 'Go straight or left', 'Turn left ahead', 'Turn right ahead'] |
| Beware of ice/snow | [9.9973279e-01 2.5119714e-04 9.9337785e-06 5.4336974e-06 2.8959482e-07] | ['Go straight or right', 'Turn right ahead', 'Ahead only', 'Roundabout mandatory', 'No entry'] |
| Wild animals crossing | [9.8058265e-01 1.8998343e-02 3.1148884e-04 4.8929003e-05 3.0442312e-05] | ['Go straight or left', 'Ahead only', 'Go straight or right', 'Turn right ahead', 'Right-of-way at the next intersection'] |
| End of all speed and passing limits | [9.3472016e-01 6.0256526e-02 2.1744212e-03 2.0546925e-03 2.1133701e-04] | ['Keep right', 'Roundabout mandatory', 'Double curve', 'No passing for vehicles over 3.5 metric tons', 'Road work'] |
| Turn right ahead | [9.9918836e-01 5.5170368e-04 1.3990254e-04 3.3782202e-05 2.1789869e-05] | ['Keep left', 'Children crossing', 'Go straight or left', 'Turn right ahead', 'Speed limit (30km/h)'] |
| Turn left ahead | [1.00000000e+00 1.40660235e-11 2.54360132e-14 1.28761654e-14 3.95320610e-15] | ['Speed limit (70km/h)', 'Speed limit (30km/h)', 'Stop', 'Speed limit (20km/h)', 'Speed limit (50km/h)'] |
| Ahead only | [0.7526928  0.23954074 0.00395806 0.00149683 0.00082426] | ['Priority road', 'Roundabout mandatory', 'Turn left ahead', 'Speed limit (100km/h)', 'Stop'] |
| Go straight or right | [9.9982148e-01 1.5955234e-04 1.7340604e-05 1.1943731e-06 3.2354271e-07] | ['End of no passing', 'End of no passing by vehicles over 3.5 metric tons', 'End of all speed and passing limits', 'End of speed limit (80km/h)', 'No passing'] |
| Go straight or left | [9.9480337e-01 4.9083265e-03 1.5550488e-04 9.1743947e-05 2.4878171e-05] | ['End of no passing', 'End of no passing by vehicles over 3.5 metric tons', 'End of speed limit (80km/h)', 'Vehicles over 3.5 metric tons prohibited', 'Slippery road'] |
| Keep right | [9.5346284e-01 4.6328969e-02 9.3569368e-05 7.7305347e-05 2.5296240e-05] | ['Speed limit (20km/h)', 'Vehicles over 3.5 metric tons prohibited', 'Speed limit (60km/h)', 'Keep left', 'Speed limit (70km/h)'] |
| Keep left | [0.7896424  0.09604284 0.02729593 0.02144158 0.02114036] | ['Children crossing', 'Pedestrians', 'End of speed limit (80km/h)', 'Keep left', 'General caution'] |
| Roundabout mandatory | [9.9491405e-01 2.6243220e-03 1.8644864e-03 4.2516852e-04 1.4913223e-04] | ['Speed limit (100km/h)', 'Speed limit (120km/h)', 'Speed limit (80km/h)', 'Speed limit (30km/h)', 'Speed limit (20km/h)'] |
| End of no passing | [9.62902486e-01 1.66855920e-02 1.31839905e-02 7.08994223e-03 9.74895011e-05] | ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (70km/h)', 'Speed limit (120km/h)', 'Speed limit (60km/h)'] |
| End of no passing by vehicles over 3.5 metric tons | [0.93091583 0.04698668 0.01068467 0.00790273 0.0028312 ] | ['End of no passing', 'End of all speed and passing limits', 'No passing', 'Vehicles over 3.5 metric tons prohibited', 'Children crossing'] |



