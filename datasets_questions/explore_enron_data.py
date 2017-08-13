#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print len(enron_data)

poi_count = 0
salary_count = 0
emails_count = 0
for features in enron_data.values():
	if features.has_key('poi') and features['poi'] == 1:
		poi_count += 1
	if features.has_key('salary') and features['salary'] != 'NaN':
		salary_count += 1
	if features.has_key('email_address') and features['email_address'] != 'NaN':
		emails_count += 1

print "poi count:", poi_count
print "salary count:", salary_count
print "email_address count:", emails_count