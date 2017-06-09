import csv
import numpy as np
import os.path
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from itertools import compress
import sys

def read_labels():
	data = {}
	with open("speeddateoutcomes.csv", "rb") as f:
		reader = csv.reader(f)
		next(reader, None)
		for row in reader:
			values = [int(val) if val.isdigit() else -1 for val in row]
			A = values[0]
			B = values[1]
			if values[14] != -1:
				if values[14] >= 8:
					data[(A,B)] = 1
				if values[14] <= 3:
					data[(A,B)] = 0
	return data

def read_feat(filename):
	with open(filename, "rb") as f:
		lines = f.read().split("\n")
		feat = np.zeros((len(lines) - 2, 16))
		for i in range(len(lines)-2):
			feats = [float(v) if v[0] != "-" else float("nan") for v in lines[i+1].split()[5:21]]
			feat[i] = np.array(feats)
		return np.concatenate((np.nanmean(feat, axis = 0), np.nanstd(feat, axis = 0)))


def process_data():
	X = []
	y = []
	labels = read_labels()
	for label in labels:
		filename = "features/" + str(label[0]) + "-" + str(label[1]) + ".feat"
		if os.path.isfile(filename):
			feat = read_feat(filename)
			X.append(feat)
			y.append(labels[label])
	X = 1/(1 + np.exp(np.mean(X, axis = 0) - X))
	X = np.matrix(X)
	y = np.array(y)

	return X, y

def model():
	X, y = process_data()

	m = SVC(1.0, kernel = 'linear', class_weight = 'balanced')

	print cross_val_score(m, X, y, cv = 5)
	
	m.fit(X,y)
	select = SelectFromModel(m, prefit=True)
	support = select.get_support()
	X = select.transform(X)

	print cross_val_score(m, X, y, cv = 5)

	m.fit(X,y)
	y_pred = m.predict(X)
	print np.mean(1 - np.abs(y_pred - y))
	print m.coef_

	return support

support = model()

feature_names = ["tndur", "pmin", "ptmin", "pmax", "ptmax", "pquan", "pmean", "psd", "pslope", "pslnjmp", "imin", "itmin", "imax", "itmax", "iquan", "imean"]
feature_names = [n + " mean" for n in feature_names] + [n + " std" for n in feature_names]
print list(compress(feature_names, support))

