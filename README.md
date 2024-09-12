# Real-Time_Heart_Attack_Prediction
Real-Time Heart Attack Prediction using Apache Kafka and Apache Spark

## Introduction

* The objective of this project is to develop a real-time system that can predict the likelihood of a person experiencing a heart attack using big data tools and machine learning.
* The dataset used for this project contains various features related to individuals' health
* The goal of this project is to utilize Kafka to stream data from a CSV file to Spark, where the machine learning model performed real-time predictions.

# Dataset 
The heart attack dataset used in this project The dataset has 
* 13 features
 303 instances
This dataset is open source and freely available on Kaggle. The link to the dataset is
* https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset.
The data was preprocessed to
* handle missing values
* convert categorical variables to numerical format
* scale numerical features as required

In the dataset
* 136 people no have heart disease
  * labeled as 0
* 165 people have heart disease
  * labeled as 1
![image](https://github.com/user-attachments/assets/2ef8a3f2-9c1c-40ab-9dfe-133e900cb027)

# The architecture of heart attack prediction system

![image](https://github.com/user-attachments/assets/f9f58d29-7aff-4c84-8990-eaa3c7ed046d)

# Model Generation and Storage Layer:
* The data has been loaded form CSV file into a DataFrame.
* A pipeline has been created, consists of three stages.
  * Vector assembler
  * MinMaxScalar
  * Logistic Regression Model
* Logistic Regression has been used for binary classification with following parameters:
  * maxIter = 1
  * regParam = 0.1
* Accuracy of the model is 79.59%
  
![image](https://github.com/user-attachments/assets/2d9ad6d5-8a4a-4318-a7a4-addabd9c0f30)


# Real-Time Prediction

![image](https://github.com/user-attachments/assets/c18f1747-8835-4859-86fe-90e82851c4c1)




    




