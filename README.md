# MedicalImageClassifier

## Overview

This project is to detect abnormal images from the medical images using several models trained in unsupervised learning.

## Project Structure

- src

    * source code directory.
    * clustering, feature_detection, image_processing
    
- utils

    several kinds of tool necessary for project.    

- main.py

    The main execution file

- settings
    
    several file’s path (model file, feature file, image file)
    
- requirements

    all the libraries for the project
    
## Installation and Execution of Projects

- Environment

    python 3.6, Ubuntu 18.04
    
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
     
