from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
import numpy as np
from numpy import genfromtxt, savetxt
import csv

def tokenize(dataset):
	data = []
	for x in dataset:
		sample = {}
		sample['lang'] = x[0]
		sample['owner_org'] = x[1]
		sample['following_owner'] = int(x[2])
		sample['no_of_stars'] = int(x[3])
		data.append(sample)

	vec = DictVectorizer()
	data = vec.fit_transform( data ).toarray()
	# print vec.get_feature_names()
	return data

def main():
	dataset = genfromtxt(open('train_data.csv','r'), delimiter=',', dtype=None)[1:]

	train = tokenize(dataset)
	scaler = preprocessing.MinMaxScaler()
	train = scaler.fit_transform(train)
	target = [x[4] for x in dataset]

	dataset = genfromtxt(open('test_data.csv','r'), delimiter=',', dtype=None)
	test = tokenize(dataset)
	test = scaler.transform(test)

	clf = tree.DecisionTreeClassifier()
	# clf = RandomForestClassifier(n_estimators=100)
	# clf = svm.SVC()
	clf.fit(train, target)
	result = clf.predict(test)

	dataset = dataset.astype(object)
	dataset = [dataset[i] + (result[i],) for i in range(len(dataset))]
	print dataset
	savetxt('result.csv', dataset, delimiter = ',', fmt="%s" )

if __name__ == '__main__':
	main()