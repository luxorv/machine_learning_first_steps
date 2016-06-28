#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


# read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
features = ["salary", "bonus"]
del data_dict['TOTAL']
data = featureFormat(data_dict, features)

best_salary = 0
best_name = ''

for name, value in data_dict.iteritems():
    if value['salary'] != 'NaN':
        if value['salary'] > best_salary:
            best_salary = value['salary']
            best_name = name

print(best_name, best_salary)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter(salary, bonus)

matplotlib.pyplot.xlabel('salary')
matplotlib.pyplot.ylabel('bonus')
matplotlib.pyplot.show()
