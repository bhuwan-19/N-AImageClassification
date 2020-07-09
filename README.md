# Overview

This project is to detect abnormal images from the various models trained in unsupervised learning.
We have more than 100K normal images, but only 31 abnormal images for data set. Also it is difficult to predict what abnormal image we will have in the future.
Therefore, we can't help using unsupervised learning method to detect an abnormal images because of the lack of abnormal images, much imbalance of normal and abnormal training data and the type of abnormality impossible to predict.

Currently, we have used one-class svm and isolation forest algorithm for unsupervised learning method.

## Project Structure

- data

    * model(inception, svm models and isolation forest models).
    * feature(normal, abnormal feature csv files)

- report

    There are some report files that contain the training and prediction result, and analysis of result for each unsupervised learning method.    
    
- src

    * source code directory.
    * clustering, feature_detection, image_processing
    
- utils

    several kinds of tool necessary for project.
    file management tool, OpenCV tool, feature detection tool...

- main.py

    main file to execute project.

- settings

    There are various kinds of settings in it.
    several file’s path (model file, feature file, image file) and start_tmp, end_tmp, tmp_interval, which will be explained below.
    
- requirements

    all the libraries necessary for the project execution
    
## Four Functionality of Project

- Preprocess normal and abnormal images

    * normalization and equalization the histogram of image.
    * position: /utils/cv_utils
    * /src/image_processing/image_collection
    
- Extracting features of normal and abnormal images using inception_v3.
    
    * /src/feature_detection/image_feature
    * /src/feature_detection/feature_collection
    * Since there are more than 130K normal images, it took lots of time to extract all features. Even the size of normal feature csv file is 1.6G.
    * You can check and download feature csv files in https://drive.google.com/file/d/1GgeTZpm2gT3WTCHxILucMUm8t2YgQWxb/view?usp=sharing
    
- Training one-class svm model and testing abnormality of image
    
    * Combining some of the features in data/features/train_normal_features.csv with 100% features in train_abnormal_features.csv, it is split train and test data set into the ration of 7:3.
    
    * /src/clustering/one_class_svm3.py

- Training isolation forest model using h2o and testing abnormality of image
    
    * The train data set and test data set are created newly. The train data set consists of 99% normal image features while test data set consists of 1% normal image features and 100% abnormal image features. They are created by utils/train_test_csv.py.
    At the same time, test label csv file is also created.
    
    * /src/clustering/isolation_forest.py

## Installation and Execution of Projects

- Environment

    python 3.6
    
- Installation

    ```
        pip3 install -r requirements.txt
    ```

- Execution for one class svm algorithm

    ```
        python main.py
    ```
  
  * In settings, the number of the samples of training data set can be set in range between ONE_CLASS_START_TMP and ONE_CLASS_END_TMP with the interval of ONE_CLASS_TMP_INTERVAL. 

- Execution for isolation forest algorithm
    
    ```
      python  src/clustering/isolation_forest.py
    ```
  
    * In settings, the range of n_trees(lower and upper) and it's step can be set.  