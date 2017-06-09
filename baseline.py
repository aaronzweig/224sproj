import csv
import numpy as np
import os.path
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from itertools import compress
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer

def convert(time):
	m = time%100
	t = time/100
	return str(t/60) + ":" + str((t%60)/10) + str((t%60)%10) + ":" + str(m/10) + str(m%10)

def reformat(filename):
	with open(filename, "rb") as f:
		lines = f.read().split("\n")
		fil = ["FILE NAME:  "+filename[12:-4]+"\n"]
		words = []
		start = 0
		end = 0

		for line in lines:
			data = line.split()
			if len(data) == 2:
				if end == start:
					end = int(data[0].split(":")[0])

				end += int(data[0].split(":")[1])
				words.append(data[1])
			if len(data) == 1 and len(words) > 0:
				string = convert(start) + " " + convert(end) + " " "MALE:" + "  " + ' '.join(words)
				fil.append(string)
				start = end
				words = []

		with open(filename[12:], "w") as f:
			for line in fil:
				f.write(line + "\n")

#reformat("transcripts/seekingalpha-ARNA-2016-Q2.txt")
#reformat("transcripts/seekingalpha-JCAP-2016-Q2.txt")

def read_feat(filename):
	durations = []
	transcripts = []
	with open(filename, "rb") as f:
		lines = f.read().split("\n")
		feat = np.zeros((len(lines) - 2, 16))
		for i in range(len(lines)-2):
			line = lines[i+1].split()
			start = float(line[3])
			end = float(line[4])
			durations.append(int(end*1000 - start*1000))
			transcripts.append(' '.join(line[21:]))
			feats = [float(v) if v[0] != "-" else float("nan") for v in line[5:21]]
			feat[i] = np.array(feats)
		return feat, durations, transcripts


def process_data(feat, label, test_bool):
	X, durations, transcripts = read_feat(feat)
	m = np.nanmean(X, axis = 0)
	X = np.nan_to_num(m - X)
	X = 1/(1 + np.exp(X))
	X = np.matrix(X)
	mask = [0,1,3,5,6,7,8,9,10,12,14,15]
	X = X[:, mask]

	if test_bool:
		return X, transcripts

	ts = np.load(label)
	
	start = 0
	y = np.zeros(X.shape[0])
	for i in range(X.shape[0]):
		dur = ts[start:start + durations[i]]
		y[i] = sum(dur)*1.0/(len(dur) + 0.001)
		y[i] = 1 if y[i] >= 80 else 0
		start += durations[i]

	return X, y

def cross_validate(m, X, y):
	kf = KFold(5, True)
	acc = []
	for train, test in kf.split(X):
		m.fit(X[train], y[train])
		y_pred = m.predict(X[test])
		acc.append(1 - np.mean(np.abs(y_pred - y[test])))
	return sum(acc)*1.0/len(acc)



def model():
	X, y = process_data("seek.feat", "confidence/seekingalpha-ARNA-2016-Q2.train.npy", False)
	X2, y2 = process_data("seek2.feat", "confidence/seekingalpha-JCAP-2016-Q2.train.npy", False)
	X = np.concatenate((X,X2), axis = 0)
	y = np.concatenate((y,y2))

	# baseline
	baseline = []
	for C in np.logspace(-2, 2, 10):
		m = SVC(C, kernel = 'linear', class_weight = 'balanced')
		baseline.append(cross_validate(m, X, y))

	# remove features based on speeddating models
	# keep ["tndur", "pmin", "pslope", "imin", "imax", "iquan", "imean"]
	mask = [0, 1, 6, 8, 9, 10, 11]
	X = X[:, mask]

	updated = []
	for C in np.logspace(-2, 2, 10):
		m = SVC(C, kernel = 'linear', class_weight = 'balanced')
		updated.append(cross_validate(m, X, y))

	plt.semilogx(np.logspace(-2, 2, 10), baseline, 'ro', np.logspace(-2, 2, 10), updated, 'bo')
	plt.title('Hyperparameter choice over cross-validated SVM models')
	plt.ylabel('average accuracy')
	plt.xlabel('C')
	p1 = mpatches.Patch(color='red', label='All features')
	p2 = mpatches.Patch(color='blue', label='Importance weighted features only')
	plt.legend(handles=[p1, p2])
	# plt.show()
	# plt.close()


	m = SVC(.205, kernel = 'linear', class_weight = 'balanced')
	m.fit(X,y)
	# empirically look at transcripts with high estimated confidence
	X_test, transcripts = process_data("test.feat", "", True)
	X_test = X_test[:, mask]
	y_pred = m.predict(X_test)
	pos = []
	neg = []
	for i in range(len(y_pred)):
		if y_pred[i] == 1:
			pos.append(transcripts[i])
		else:
			neg.append(transcripts[i])

	vec = TfidfVectorizer(stop_words='english')
	vec.fit_transform(pos)
	words = vec.get_feature_names()
	indices = np.argsort(vec.idf_)[::-1]
	print [words[i] for i in indices[:10]]

	vec.fit_transform(neg)
	words = vec.get_feature_names()
	indices = np.argsort(vec.idf_)[::-1]
	print [words[i] for i in indices[:10]]


model()
