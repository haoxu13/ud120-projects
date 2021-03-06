#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from time import time

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### split train/test dataset
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

### Create  Decision Tree Classifier And Validate it

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
print "Acc:", clf.score(features_test, labels_test)
pred = clf.predict(features_test)
'''
print len(pred)
# count the TP
counter = 0
for ii,ele in enumerate(pred):
	if ele == 1 and labels_test[ii] ==1:
		counter += 1
print "Test TP:", counter
'''
from sklearn.metrics import precision_score, recall_score
print "precision:", precision_score(labels_test, pred)
print "recall:", recall_score(labels_test, pred) 


