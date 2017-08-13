#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL')
data = featureFormat(data_dict, features)


### your code below

### find the outlier with the most salary
most_salary = 0
most_salary_name = ''
for _name, _features in data_dict.iteritems():
	salary = _features['salary']
	bonus = _features['bonus']
	if salary > most_salary and  salary != 'NaN':
		most_salary = salary
		most_salary_name = _name
	if salary > 1000000 and  salary != 'NaN':
		print "name:", _name, " salary:", salary, " bonus:", bonus

print "most salary name:", most_salary_name
print "salary:", most_salary

### visualization
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()