# Real-Time-Face-Mask-Detection

## SSDMNV2: Single Shot Multibox Detector and MobileNetV2
Developed lightweight MobileNetV2 face mask detection model for identifying a person wearing a mask or not with an accuracy of 92.64% and f1 score of 0.93%.

## Demonstration of Project:
<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123558600-c7235a00-d7b4-11eb-876d-73d1ec19b1bf.gif" />
</p>


### Link for Research Paper: https://www.sciencedirect.com/science/article/pii/S2210670720309070#bib0195
### Contributed in other researches: https://scholar.google.com/citations?user=73b_WZcAAAAJ&hl=en
### Link to Download Complete Dataset: https://github.com/TheSSJ2612/Real-Time-Medical-Mask-Detection/releases/download/v0.1/Dataset.zip

## How to use?
1. Download the required files into a directory if your choice.
2. Open required codes in Jupyter Notebook.  
3. Install the dependencies as mentioned in code.
4. Execute the code.
5. Press 'q' to exit from real time video detection.
6. Done

## Table of Contents: 
1. [Abstract](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#1-abstract)
2. [Motivation](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#2-motivation)
3. [Problem Statement](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#3-problem-statement)
4. [Introduction](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#4-introduction)
5. [Requirements](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#5-requirements)
6. [Dataset Creation](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#6-dataset-creation)
7. [Generic Methodology](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#7-generic-methodology)
8. [Results](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#8-results)
9. [Comparison with other model](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#9-comparison-with-other-model)
10. [Summary and Conclusion](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#10-summary-and-conclusion)
11. [Limitations](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#11-limitations-challenges-faced-during-the-project)
12. [Future Scope](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#12-future-scope)
13. [Credits](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#13-credits)
14. [License](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection#14-license)

## 1. Abstract
- This project is concerned with the detection of face masks.
- There are two phrases of our project:
  - Training: Detect face mask on images (using Keras/TensorFlow)
  - Deployment: Detect face mask on real  time video stream
- Dataset consist of  11,042 images combined manually.
- MobileNetV2 architecture is used for fine tuning on the dataset.
- Adam is used as optimizer and Binary Cross Entropy is used as a loss function.
- Our proposed model gives an accuracy of 92.53%.

## 2. Motivation
- In the rapid developing world, with increase in technology so is increasing diseases such as Covid-19.
- To prevent the spread of disease it is absolutely compulsory to wear mask.
- There is a need for Mask Detection technique as checking mask for every person by a person is not feasible, therefore we need a deep learning algorithm which will ease the work of human workforce.


## 3. Problem Statement
- The goal is to predict whether the person is wearing a mask or not.
- Since this is an issue of binary classification, the model if predicts correctly then a green rectangular box appears on person wearing mask. 
- If the person is not wearing mask, the model predicts red rectangular box on his face along with accuracy score.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123409678-135f7600-d5cc-11eb-8fe7-864357267a8c.png" />
</p>
<p align=center> 
Figure 1: Mask Found
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123409982-6df8d200-d5cc-11eb-80a2-b7fe603b2c39.png" />
</p>
<p align=center> 
Figure 2: Mask not found
</p>

## 4. Introduction
- Face Mask detection has become a very trending application due to Covid-19 pandemic, which demands for a person to wear face masks, keep social distancing and use hand sanitizers to wash your hands. 
- While other problems of social distancing and sanitization have been addressed up till now, the problem of face mask detection has not been addressed yet.
- This project proposes a model for face detection using OpenCV DNN, TensorFlow, Keras and MobileNetV2 architecture which is used as an image classifier. 
- This dataset could be used for developing new face mask detectors and performing several applications.

## 5. Requirements
All the experimental trials have been conducted on a laptop equipped by an Intel i7-8750H processor (4.1 GHz), 16 GB of RAM with 1050ti max-Q with 4 GB of VRAM. 
The Jupyter Notebook software equipped with Python 3.8 kernel was selected in this project for the development and implementation of the different experimental trails.

- Jupyter Notebook version 6.1 or above
- Python version 3.8 or above
- Python Libraries Used:
  - numpy https://numpy.org/doc/
  - pandas https://pandas.pydata.org/docs/
  - TensorFlow https://www.tensorflow.org/api_docs
  - MobileNetV2 https://keras.io/api/applications/mobilenet/
  - OpenCV https://docs.opencv.org/3.4/
  - scikit-learn https://scikit-learn.org/stable/user_guide.html
  - keras https://keras.io/guides/
  - matplotlib https://matplotlib.org/stable/users/index.html
  - seaborn https://seaborn.pydata.org/tutorial.html

## 6. Dataset Creation

### Sources of Hindi Dataset Creation
1. Kaggle's Medical Mask Dataset by Mikolaj Witkowski: https://www.kaggle.com/mloey1/medical-face-mask-detection-dataset
2. Masked face recognition dataset and application: https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
3. Prajna Bhandary dataset available at PyImageSearch: https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/

### Dataset Description

<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123553442-d2688c80-d798-11eb-9752-35c3f5031a7d.png"/>
</p>
<p align=center> 
Figure 3: Bar graph with class: 
"with mask" and "without mask"
</p>

- The data set consist of 11,042 total images out of which 80 percent of images used for training  data set, and the rest 20 percent have been used for testing.
- The following figure 3 shows the bar graph which represents equal 5521 images distribution between two classes, with_mask  and without_mask.
- This accounts for approximately 8,833 images and 2,209 images, which have been used for training and testing respectively. 
- There is equal distribution of dataset.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123553223-908b1680-d797-11eb-89f3-9fe95d48bb9e.png" />
</p>
<p align=center> 
Figure 4: Dataset 25 images
</p>

- The figure 4 represents 25 images choosen at random from complete dataset. 
- The following figure 4 shows dataset description as follows:
  - 'with_mask' representing image with person wearing mask.
  - 'without_mask' representing image with person not wearing a mask.

## 7. Generic Methodology
<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123554245-abac5500-d79c-11eb-9d9e-dfc0df0ebc9b.png" />
</p>
<p align=center> 
Figure 5: Data Pipeline
</p>

- The following figure 5 represents sequential steps performed in order to reach to end goal.
- To predict whether a person has worn a mask correctly, the initial stage would be to train the model using a proper dataset.
- After training the classifier, an accurate face detection model is required to detect faces, so that the SSDMNV2 model can classify whether the person is wearing a mask or not.
- The task in this project is to raise the accuracy of mask detection without being too resource-heavy.
- For doing this task, the DNN module was used from OpenCV, which contains a ‘Single Shot Multibox Detector’ (SSD) object detection model with ResNet-10 as its backbone architecture.
- This approach helps in detecting faces in real-time, even on embedded devices like Raspberry Pi.
- The following classifier uses a pre-trained model MobileNetV2 to predict whether the person is wearing a mask or not.

## 8. Results

### Confusion Matrix of SSDMNV2:
<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123554750-28403300-d79f-11eb-9b76-b1dbfacd92c4.png" />
</p>
<p align=center> 
Figure 6: Heatmap Respresenting Confusion Matrix
</p>

- The confusion matrix is plotted with help of heatmap showing two dimensional matrix data in graphical format.
- It has successfully identified 941 true positives, 1103 true negatives, 2 false positive and 163 false negatives.

### Training accuracy/loss curve on train and validation dataset

<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123554790-5faedf80-d79f-11eb-8f65-692e3e5d2589.png" />
</p>
<p align=center> 
Figure 7: Training accuracy/loss curve on train and validation dataset
</p>

- The plots are based on model accuracy/loss, the pyplot command style function that makes matplotlib work like Matlab.
- In this figure the violet curve shows the training accuracy which is nearly equal to 98%, the grey curve represents training accuracy on the validation dataset.
- Training loss where the red curve shows loss in training dataset less than 0.1, whereas the blue curve shows training loss on the validation dataset.

## 9. Comparison with other model

<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123554877-dea41800-d79f-11eb-900b-6babdc18ac1c.png" />
</p>
<p align=center> 
Figure 8: Comparison with other models
</p>

###  Comparison with other implementation
<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123554923-20cd5980-d7a0-11eb-9f21-2ecea38093cd.png" />
</p>
<p align=center> 
Figure 8: Comparison with other implementations
</p>

#### The following figure 8 represents on the left predictions made by similar methodology used by PyImageSearchour model vs on the right predictions made by our SSDMNV2 model.


## 10. Summary and Conclusion
- In our face mask detection model we successfully performed both the training and development of the image dataset which were divided into categories of people having masks and people not having masks.
- We were able to classify our images accurately using MobileNetV2 image classifier which is one of the uniqueness of our model.
- The technique of Object detection using OpenCV deep neural networks in our approach generated fruitful results.
- The real time face detection model has an accuracy of 92.53% and produces a highest F1 score with 0.93.
- A successfull research paper has been published in Elsivier journal in Sustainanble societ


## 11. Limitations/ Challenges faced during the project
- The collection of the labeled dataset was a problem because of the unavailability of the properly labeled dataset.
- Preprocessing of data was a challenge since the dataset from Masked face recognition and application contained a lot of noise, and a lot of repetitions were present in the images of this dataset.
- Finding these corrupt images was a tricky task, but due to valid efforts, we divided the work and cleaned the data set with mask images and without mask images.
- Due to the non-availability of an adequate amount of data for training the proposed model, the method of data augmentation is used to solve this issue.
- The results were analyzed before applying augmentation and after applying augmentation.
- The time taken for training the model took a lot of time (~6 to 7 hours), which was enhanced with the help of the Nvidia GPU for faster processing and calculations.

## 12. Future Scope
- To increase the size of dataset and make it more robust.
- To investigate new features to improve existing model.
- Incorporating model into raspberry pi for real time identification.
- Other researchers can use the dataset provided in this paper for further advanced models such as those of face recognition, facial landmarks, and facial part detection process.

## 13. Credits: 
Thanking my project teammates for caring and supporting me wholeheartedly. The role you played in my life is invaluable. I’m grateful for all of your help and continued support.
<div class="align-text">
  <p>
   <p text-align= "justify"> Agam Madan : https://www.linkedin.com/in/agam-madan/  </p>   
   <p text-align= "justify"> Rohan Arora : https://www.linkedin.com/in/rohanarora18/  </p> 
  </p>
</div>

## 14. License: 
- Apache License 2.0
