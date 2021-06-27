# Real-Time-Face-Mask-Detection

## SSDMNV2: Single Shot Multibox Detector and MobileNetV2
Developed lightweight MobileNetV2 face mask detection model for identifying a person wearing a mask or not with an accuracy of 92.64% and f1 score of 0.93%.

## Table of Contents: 
1. Abstract
2. Motivation
3. Problem Statement
4. Introduction
5. Requirements
6. Dataset
7. Generic Methodology
8. Results
9. Comparison with other model
10. Conclusion
11. Limitations
12. Future Scope
13. Credits
14. License

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
  - scikit-learn https://scikit-learn.org/stable/user_guide.html
  - matplotlib https://matplotlib.org/stable/users/index.html
  - seaborn https://seaborn.pydata.org/tutorial.html

## 5. Dataset Creation
### Sources of Hindi Dataset Creation
1. Kaggle's Medical Mask Dataset by Mikolaj Witkowski https://www.kaggle.com/mloey1/medical-face-mask-detection-dataset
2. Prajna Bhandary dataset available at PyImageSearch https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
3. Masked face recognition dataset and application

The complete dataset is available @ https://github.com/TheSSJ2612/Real-Time-Medical-Mask-Detection/releases/download/v0.1/Dataset.zip

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

- The following figure 4 shows dataset description as follows:
  - 'with_mask' representing image with person wearing mask.
  - 'without_mask' representing image with person not wearing a mask.

## 6. Generic Methodology
<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123554129-448ea080-d79c-11eb-9906-1ad4f0ffc552.png" />
</p>
<p align=center> 
Figure 5: Flow Diagram of the SSDMNV2 model
</p>

- The following figure represents sequential steps performed in order to reach to end goal.

## 7. Comparison of Results
<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123553932-68051b80-d79b-11eb-8f89-615111b84cba.png" />
</p>
<p align=center> 
Figure 6: Comparison of accuracy between different models
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123554027-cfbb6680-d79b-11eb-8800-0e0543be3891.png" />
</p>
<p align=center> 
Figure 7: Comparison of F1 Score between different models
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123553994-aef31100-d79b-11eb-94a8-201ca02c749d.png" />
</p>
<p align=center> 
Figure 8: Comparison of Performance between different models using FPS parameter
</p>

In the end, MobileNetV2 was chosen to be our model for the proposed approach since it outperforms other models in F1-score and Average performance. This makes the proposed approach easy to deploy in real-time even on embedded devices which is not possible with heavy models and to do real-time detection using these models requires good computational power which might make it difficult to play in real life.

### Architecture used in our model
<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123116562-a37fad00-d45e-11eb-9b8e-db4e85dfb1d7.png" />
</p>
<p align="center">
  Bidirectional LSTM Architecture retrieved from https://paperswithcode.com/method/bilstm
</p>

- It is a two way process.
- A Bidirectional LSTM is a sequence processing model that consists of two LSTMs: one taking the input in a forward direction, and the other in a backwards direction. 
### Example of B-LSTM:
<p align="center">
  <img src="https://user-images.githubusercontent.com/47279598/123117182-299bf380-d45f-11eb-8ee4-b278f5c8d8b8.png" width="800" height="400" />
</p>

- The LSTM model reads the input text in one direction from left to right.
- The B-LSTM model reads the input text from both directions from left to right and right to left.

## 8. Summary and Conclusion
- Automatic fake news detection is a very promising area of research.
- Due to drastic consequences detection of fake news becomes very significant. 
- The Hindi dataset created can be a contribution to other research work. 
- The project proposes a model that can easily absorb other features of news and has a very strong extensibility. 
- B-LSTM was preferred since higher accuracies were achieved of about 95.01%.


## 9. Limitations/ Challenges faced during the project
- Lack of labelled data availability in Indian regional languages.
- The amount of data on social media is massive but unlabeled and hence could not be used for training.
- Also preprocessing of Hindi data was a challenge.
- Due to above limitations remaining available dataset will lead to underfitting of the model.

## 10. Future Scope
- To increase the size of Hindi dataset and make it more robust.
- Testing the model using URL to validate headlines and other parameters.
- To make system adaptive to other languages and detect region specific biases.
- To investigate new features to flag fake news.

## 11. Credits: 
Thanking my project teammates for always inspiring and motivating me throughout the journey.
<div class="align-text">
  <p>
    <p text-align= "justify"> Rohan Arora : https://www.linkedin.com/in/rohanarora18/  </p> 
    <img src="https://user-images.githubusercontent.com/47279598/123132503-70441a80-d46c-11eb-9157-47d93081864d.png" align="justfy" width="250" height="250"/>
    <p text-align= "justify"> Shreya Dhingra https://www.linkedin.com/in/shreya-dhingra-927b19190 </p>  
    <img src="https://user-images.githubusercontent.com/47279598/123132674-9c5f9b80-d46c-11eb-9e0e-5c7d716fb811.png" align="justify" width="250" height="250"/>
    <p text-align= "justify"> Adarsh Kumar https://www.linkedin.com/in/adarsh-kumar-5b1a1719b </p>  
    <img src="https://user-images.githubusercontent.com/47279598/123133053-fc564200-d46c-11eb-8b87-cea10612aee1.png" align="justfy" width="250" height="250"/>
  </p>
</div>

## 12. License: 
- Apache License 2.0
