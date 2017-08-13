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

### Create Overfit Decision Tree And Validate it

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features, labels)
score = clf.score(features, labels)
print "Acc:", score

### Using k-fold Cross Validate
'''
from sklearn.model_selection import cross_val_score
t0 = time()
scores = cross_val_score(clf, features, labels, cv=5)
print "training time:", round(time()-t0, 3), "s"
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
'''