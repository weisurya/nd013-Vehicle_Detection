## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/ratio.png
[image2]: ./images/sample-hog-output.jpg
[image3]: ./images/sample-hog-output2.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

All of the code for this project was put up together at `Vehicle-Detection.ipynb`. Most of this function was similar with the code from the exercise. I just re-write it in order to perceive a better understanding of each of it, create e function testing, and tune the hyperparamter.

First of all, I start by loading the required images, which is `test_images` from the test images that was provided. I also use the labelled dataset from Udacity. The `vehicles`, and `non_vehicles` dataset was downloaded through [this link for the vehicle dataset](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [this link for the non-vehicle dataset](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). Because the dataset is provided, I had had no idea whether the distribution between vehicle images and non-vehicle images have been balanced. So, I need to check it first to avoid imbalanceness.

![alt text][image1]

The way I extract HOG features could be seen at `get_hog_features()`, which I need to specify `pix_per_cell` and `cell_per_block` beforehand. This function are used by being called from either `extract_hog_features()` which is only using HOG as the only feature, or `extract_features()`, by combining color and HOG features together. In this project, I'm using `extract_features()` to extract the features.

I try various hyperparameters combination, and I found that using `color_space = 'LUV'`, `orient = 8`, `pix_per_cell = 8`, `cell_per_block = 2`, `hog_channel = 0`, `spatial_size = (16, 16)`, `hist_bins = 32` could deliver quite satisfying result with 97.94% of test accuracy by using linear SVM. Here is the example output from test image:
.
![alt text][image2]

As we see, there are a lot of false positives. I tackle this issue in the 6th section of my notebook, `Heat-map Detection for False Positive Issue`


#### 2. Explain how you settled on your final choice of HOG parameters.

I try various hyperparameters combination, and I found that using `color_space = 'LUV'`, `orient = 8`, `pix_per_cell = 8`, `cell_per_block = 2`, `hog_channel = 0`, `spatial_size = (16, 16)`, `hist_bins = 32` could deliver quite satisfying result with 97.94% of test accuracy by using SVM. I use it because it can provide me an output that I was desired, which give me a ton of false positives in the targeted object and less false positives in non-targeted object which I could tackle it later on by using heatmap and thresholding.
.
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using default hyperparameter by using the dataset that I have combined and normalized through these function: `combine_features()`, `fit_scaler()`, `apply_scaler()`, and `split_train_test_set()`. I am using 80% of total data as the training set and 20% of total data as test set. This setting could deliver a satisfying result with 97.94% of test accuracy.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I create several functions to answer this section, `slide_window()`, `draw_boxes()`, `single_img_features()`, and `search_windows()`. The `single_img_features()` is similar with `extract_features()`, which this function only extract features from a single image rather than a bunch of images. It was used to support `search_windows()`.

In `slide_window()`, I need to set `x_start_stop`, `y_start_stop`, `xy_window` and `xy_overlap`. I am tune the `y_start_stop` in order to determine on which are I need to slide the window to search the desired object, and I set it from 400 until 656 px. After that, I set the `xy_overlap` into 0.85 in order to retrieve a lot of false positives for the targeted object for heatmap and thresholding problem.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I find that tuning on `y_start_stop` helps me to improve the speed performance because I don't need to search obkect from the whole image. I also tune `xy_overlap` into 0.85 because it could give me an adequeate false positives on specific object that I desired, which I could use it for thresholding on the heatmap section. Here is an example:

![alt text][image3]

I optimize these section by tuning both hyperparameters manually.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/mfY-QJo-SXU) and [here](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

From the previous section, `5. Search & Classify Testing`, I continue it by creating several new function for heatmap and thresholding, `add_heat()`, a function that add "heat" to a map for a list of bounding boxes, `apply_threshold()`, a function that apply thresholding method for heatmap, and `draw_labeled_bboxes()`, a function that wrap both to function before together to get a new labeled box.

By getting a lot of false positives for the targeted object, which is car, and less false positives for the non-targeted object, which is non-car, I could use this gap to apply thresholding method. First of all, I convert the detected object into heat on heatmap. Then, I set if the heat is below the nominal of threshold that I define, I would be sorted out.

In the next section, `7. Lane Line`, I only put all of the functions that I created for the previous project, and use it for the next section, `8. Video Pipeline`. In this section, I create a pipeline function, `draw_lane_and_vehicle_tracking()`, to combine lane finding and vehicle tracking functions together. This function also be used for the final process, which is to create the video output.


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Actually, this project is quite straightforward, which I could use several methods that was taught from the lesson and put it up together. Several challenges that I found were:

* I created a function testing for every function that I created. The problem was I use a common variable name for testing purpose, which similar variable name also being used for development purpose. This created several anomalies which took a lot of times to trace.
* I think I could reduced the false positive by creating another layer for thresholding, in order to reduce the non-vehicle images being detected. Maybe another prediction method could be applied, which is using YOLO method.