# YoCol
Implementation of YOLOv3 with opencv and color classifier with KNN based on color histogram in python 3.
---

## Introduction
To response the challenge of recognizing car make, model, and color from aiforsea.com, i propose method using YOLO v3 to detect car make and model, then croping object from image based on bounding box and passing it into color classifier. This repository implement darknet from [YOLO v3 repository](https://github.com/pjreddie/darknet) to train the model, and color classifier from [color-classifier](https://github.com/ahmetozlu/color_classifier) to classify its color. For detection, i use opencv to read the configuration and weights files, forwarding image to model, and get the bounding box of the object. This implementation is used python programming language for inference and C programming languange for darknet (training process)

### YOLO v3 Object Detection
As you can see, YOLO v3 is object detection that can detecting object from given image with one stage scanning process over the image. It split a given image into grid and perform Convolution for each of grid to extract the features and classify object. Ouput from this network is coordinate of bounding box of detected object.

In this implementation, i decide to add 196 classes of given dataset (car make and model) to pretrained weights file that can recognize 80 classes from COCO dataset. Before passing train images to YOLO, i labeling again the train images that have width upper than 300 pixel

### Color Classifier
This color classifier focus on classify ten color that commonly used in a car. The color consist of black, blue, brown, gray, green, red, silver, white, and yellow. Color are classified by using K-Neares Neşghbor Machine Learning classifier algorithm. This classifier is trained by image R, G, B Color Histogram values. The general work flow is given at the below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/35335133-a9632c70-0125-11e8-9204-0b4bfd0702a7.png" {width=35px height=350px}>
</p>

There is 2 main method to classify color in this implementation: 

**1.) Feature Extraction** = Color Histogram

Color Histogram is a representation of the distribution of colors in an image. For digital images, a color histogram represents the number of pixels that have colors in each of a fixed list of color ranges, that span the image's color space, the set of all possible colors.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/34918867-44f5feaa-f96b-11e7-9994-1747846266c9.png">
</p>

**2.) Classification** = K-Nearest Neighbors Algorithm

K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions). KNN has been used in statistical estimation and pattern recognition already in the beginning of 1970’s as a non-parametric technique.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/34918895-c7b94d24-f96b-11e7-87da-8619d9bd4246.png">
</p>
---
## Dependencies
Some dependencies that make YoCol work : 
- For inference, because we use OpenCV to read YOLOv3, we aren't use CUDA or CuDNN
  - Numpy
  - Matplotlib
  - Scipy
  - OpenCV-python
- For training, you can see the dependencies mentioned in .ipynb file
---

## Installation
```Shell
$ git clone https://github.com/Oskop/YoCol.git
$ cd YoCol
$ pip install -r requirement.txt
```
The directories of this repository will be like :
```Shell
   YoCol
   |-darknet
   |---cfg
   |---data
   |-----labels
   |---examples
   |---include
   |---python
   |---scripts
   |---src
   |-data
   |---color_training_dataset
   |-----black
   |-----blue
   |-----green
   |-----orange
   |-----red
   |-----violet
   |-----white
   |-----yellow
   |-utils
   |---color_recognition_api
   |-----__pycache__
```
---

## Usage
Before you want to use this implementation, you must download the weights file [availabel soon](). Put the model into ```data``` directory. Then you can run this command bellow :
```
usage: yocol.py [--image]
                [--video]

optional arguments:
  --image IMAGE      path to image file.
  --video VIDEO      path to video file. .mp4 format is recommended
without arguments:
  it will use your webcam as input
```
---

## Training
### For Object Detection
To train object detection model, you can use .ipynb file and run on Colab. All configuration is settled in that file. If you want to train the model with your own dataset, follow the instruction from [YOLO-Annotation-Tool-New](https://medium.com/@manivannan_data/yolo-annotation-tool-new-18c7847a2186) written by [Manivannan Murugavel](https://medium.com/@manivannan_data) for labelling.

### For Color Classifier
To train color classifier, just put color images in "data/color_training_dataset/\<color\>" with name format "ColornameNumber" (ex. blue17.jpg) with number start from last number in <color> directory. Then run this command :
```Shell
$ python train_color.py
```
it will create \<training.data\> to "data" directory

## Evaluation
Will be available as soon as possible
