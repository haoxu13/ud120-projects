#!/usr/bin/python

'''
    0. preprocessing data (scale, outlier, make new features)
    1. what feature and how many features and why (just try choose n from m, n in range(m))
    2. what algorithm and what parameters (define as many as possible)
    So the best solution will be:
        grid search:
            feature x algorithm

    score is defined as f1 score

'''

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from time import time


### candidate features
features_list = (['poi','salary', 'to_messages', 'deferral_payments', 'total_payments', 
    'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 
    'restricted_stock_deferred','total_stock_value', 'expenses', 'loan_advances', 'from_messages', 
    'other', 'from_this_person_to_poi','director_fees', 'deferred_income', 'long_term_incentive', 
    'from_poi_to_this_person'])

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers and Handle 'NaN'
### outliers is observed from scatter plot. see file enron_outliers.py
data_dict.pop('TOTAL')

for _name, _features in data_dict.iteritems():
    for _feature_name, _value in _features.iteritems():
        if _feature_name != 'poi' and _value == 'NaN':
            _features[_feature_name] = 0


### Create new features:
### to_mail / from_mail , from_this_person_to_poi / from_poi_to_this_person
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Preprocessing the data
from sklearn.preprocessing import MinMaxScaler
features = MinMaxScaler().fit_transform(features)

### Select k-best features, K is selected by grid search
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

for k in range(1,20):
    print "k:", k
    selector = SelectKBest(f_classif, k).fit(features, labels)
    new_features = selector.transform(features)
    new_features_list = []
    for _feature_name, _mask in zip(features_list[1:], selector.get_support()):
        if _mask == True:
            new_features_list.append(_feature_name)
    print "new_features_list:", new_features_list

    clf = SVC()
    cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( new_features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( new_features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


