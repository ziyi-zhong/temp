from sklearn.neural_network import MLPClassifier
import json, numpy as np, pickle, sys, os

FEATURE_PATH = "./mfcc_kmeans_200"

X_trn, X_val, Y_trn, Y_val = [], [], [], []

f = open("trn.lst", "r")
trn_files = [x.strip('\n') for x in f.readlines()]
f.close()
f = open("val.lst", "r")
val_files = [x.strip('\n') for x in f.readlines()]
f.close()

player_info = json.load(open("num_of_followers.json"))

for fname in trn_files:
	user = fname.split()[0].strip()
	feat_file = os.path.join(FEATURE_PATH, fname+".fkmeans")
	feat = pickle.load(open(feat_file, "rb"), encoding='latin1')
	assert feat.shape == (200,)
	X_trn.append(feat)
	Y_trn.append(player_info[user]['cluster'])

for fname in val_files:
	user = fname.split()[0].strip()
	feat_file = os.path.join(FEATURE_PATH, fname+".fkmeans")
	feat = pickle.load(open(feat_file, "rb"), encoding='latin1')
	assert feat.shape == (200,)
	X_val.append(feat)
	Y_val.append(player_info[user]['cluster'])

X_trn = np.array(X_trn)
Y_trn = np.eye(10)[Y_trn]
X_val = np.array(X_val)
Y_val = np.eye(10)[Y_val]

clf = MLPClassifier()
clf.fit(X_trn, Y_trn)
print (clf.score(X_val, Y_val))

pickle.dump(clf, open("mlp", 'wb'))

