#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

# Measure the time before starting the learning.
t0 = time()
# Train the classifier with the data
classifier.fit(features_train, labels_train)
# Print the time spent on the training
print "Training time:", round(time()-t0, 3), "s"

# Measure the time before starting the prediction
t0 = time()
# Predict the classification of the new data
classifier.predict(features_test)
# Print the time spent on the prediction.
print "Prediction time:", round(time()-t0, 3), "s"
#########################################################


