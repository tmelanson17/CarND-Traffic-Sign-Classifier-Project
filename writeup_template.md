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

[image1]: ./output/train_histogram.png "Frequency Count of Traffic Sign IDs"
[image2]: ./output/sample_images.png "Grayscaling"
[image3]: ./output/5_images.png "Color Images"
[image4]: ./test_images/000001.jpg "Traffic Sign 1"
[image5]: ./test_images/000002.jpg "Traffic Sign 2"
[image6]: ./test_images/000003.jpg "Traffic Sign 3"
[image7]: ./test_images/000004.jpg "Traffic Sign 4"
[image8]: ./test_images/000005.jpg "Traffic Sign 5"
[image9]: ./output/image_examples.png "Visualization of Each Label (from Training Dataset)"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/tmelanson17/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python len() function plus numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a diagram visualizing each data label in the dataset, followed by the frequency of each label in the training dataset.

![alt text][image9]
![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because sign shape and content were more important than the color of the traffic sign. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]
![alt text][image2]

As a last step, I normalized the image data so the image inputs wouldn't be affected by an unnecessary bias (regular images have an average value around 128) and would be within a resonable bounds so the output wouldn't explode. This normalization step actually results in significant improvement of the overall data. 

Adding additional data proved to be a challenge, as generating it would become more than my system RAM could handle.

Additionally, I attempted to rotate the images by a random amount. As reversing an image may change the sign label or render it unreadable, this step makes the most intuitive sense for increasing spatial robustness. However, even a slight rotation of 10 degrees resulted in worse overall data performance.

[//]: <> I decided to generate additional data because ... 
[//]: <> To add more data to the the data set, I used the following techniques because ... 
[//]: <> Here is an example of an original image and an augmented image:


[//]: <> The difference between the original data set and the augmented data set is the following 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6       				|
| Flatten               | outputs 400                                   |
| Fully connected		| outputs 120        							|
| Dropout       		| prob = 0.5        							|
| Fully connected		| outputs 84        							|
| Dropout       		| prob = 0.5        							|
| Fully connected		| outputs n\_classes (43)        				|
| Softmax				|           									|
| Cross-Entropy Loss	|												|
|						|												|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with a batch size of 128 for 20 epochs. The learning rate was set to 0.001, mu=1.0, sigma=0.3, dropout probability=0.5.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.951
* test set accuracy of 0.937

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
LeNet was chosen, as it was the best model for small inputs
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 The lack of dropoff between train, validation, and test accuracy suggests that the model is somewhat robust to mistakes (though still somewhat overfitting)

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (30 km/h) | Yield        									| 
| Turn Left Ahead     	| Turn Left Ahead 								|
| Priority Road			| Priority Road									|
| Road Work	      		| Road Work 					 				|
| General Caution	    | General Caution      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure that this is a yield sign (probability of 0.98), and the image does not contain a stop sign. In fact, the The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .998         			| Priority   									    | 
| .002                  | Keep Right                        |
| .000                  | Yield                        |
| .000                  | Turn Left Ahead                        |
| .000                  | Go Straight or Right                        |


For the last few images, the model predicted correctly, with an accuracy of nearly 100% (all other options had >.00001 probability). Although this is a clear example of overfitting, the model did very well given the lack of previous exposure to the data.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00     				| Turn Left Ahead 								|
| .000                  | Ahead Only                        |
| .000                  | Yield                        |
| .000                  | Turn Right Ahead                        |
| .000                  | Go Straight or Left                        |


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00					| Priority Road									|
| .000                  | Roundabout Mandatory                        |
| .000                  | Stop                        |
| .000                  | Speed Limit (50 km/h)                        |
| .000                  | End of No Passing                     |


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00	      			| Road Work					 				    |
| .000                  | Dangerous Curve to the Left                        |
| .000                  | Right-of-Way to the Next Intersection                        |
| .000                  | General Caution                        |
| .000                  | Pedestrians                        |


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00				    | General Caution     							|
| .000                  | Right-of-Way to the Next Intersection                        |
| .000                  | Pedestrians                        |
| .000                  | Traffic Signals                        |
| .000                  | Go Straight or Left                        |



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


Several activations highlighted the edge around the sign. This was probably adapted to distinguish signs by their shape. For example, on my speed limit sign, there was a triangular patch of woods in the background. This was caught by one of the earlier activation layers and was likely used to determine the presence of the Yield sign. 

Additionally, any distinguishable markings on the sign, such as arrows or exlamation points, were activated in different layers.

Finally, for the yield sign, one layer showed higher activations on whole regions; for example, the Priority Road sign's yellow interior was highlighted, as it was a notable feature even in black-and-white images.