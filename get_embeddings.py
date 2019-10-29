import keras
import keras.backend as K
import numpy as np, json, pickle, time, sys, os, codecs
from sklearn.svm.classes import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score

FEATURE_PATH = "./mfcc_kmeans_200/"
ALL_FILES = "./audio_features/"

def normalize(vec):
	vec = np.array(vec)
	norm = np.linalg.norm(vec)
	res = vec / norm
	assert vec.shape == res.shape
	return res

def get_player_rank():
	return json.loads(open("player_rank.json", "r").readline())

def triplet_loss(inputs):
	"""
	Using square Euclidean distance and maxplus margin.
	"""
	anchor, pos, neg = inputs
	pos_dist = K.square(anchor - pos)
	neg_dist = K.square(anchor - neg)

	# Euclidean Distance
	pos_dist = K.sum(pos_dist, axis=-1, keepdims=True)
	neg_dist = K.sum(neg_dist, axis=-1, keepdims=True)

	# Softplus Margin
	loss = pos_dist - neg_dist
	# return K.mean(K.log(1 + K.exp(loss)))
	return K.mean(K.maximum(0.0, 1. + loss))

def get_mlp(input_shape=(200,)):
	mlp = keras.Sequential()
	mlp.add(keras.layers.Dense(units=100, input_shape=input_shape, activation='sigmoid', name='dense_1'))
	mlp.add(keras.layers.Dropout(0.1))
	mlp.add(keras.layers.Dense(units=50, activation='sigmoid', name='dense_2'))

	return mlp

def get_model(input_shape=(200,)):
	mlp = get_mlp(input_shape)
	anchor_input = keras.layers.Input(input_shape, name='anchor_input')
	pos_input = keras.layers.Input(input_shape, name='pos_input')
	neg_input = keras.layers.Input(input_shape, name='neg_input')

	anchor_embedding = mlp(anchor_input)
	pos_embedding = mlp(pos_input)
	neg_embedding = mlp(neg_input)

	inputs = [anchor_input, pos_input, neg_input]
	outputs = [anchor_embedding, pos_embedding, neg_embedding]

	triplet_model = keras.models.Model(inputs, outputs)
	triplet_model.add_loss(K.mean(triplet_loss(outputs)))

	return mlp, triplet_model


def get_data(lst, ranks):
	f = open(lst, "r")
	file_lst = [x.strip('\n') for x in f.readlines()]
	f.close()
	X = [pickle.load(open(os.path.join(FEATURE_PATH, x + ".fkmeans"), "rb"), encoding='latin1') for x in file_lst]
	Y = [ranks[x.split()[0].strip()] for x in file_lst]
	return np.array(X), np.array(Y)


def embed(X, model):
	inputs = {'anchor_input': X, 'pos_input': np.zeros(X.shape), 'neg_input': np.zeros(X.shape)}
	anchor_embedding, _, _ = model.predict(inputs)
	return anchor_embedding


if __name__ == '__main__':
	mlp, triplet_model = get_model()
	triplet_model.load_weights('triplet_pre_40.best.hdf5')
	ranks = get_player_rank()
	X_trn, Y_trn = get_data("trn_esea_real.lst", ranks)
	X_val, Y_val = get_data("val_esea_real.lst", ranks)

	trn_embedding = embed(X_trn, triplet_model)
	val_embedding = embed(X_val, triplet_model)

	# print (X_trn, X_val)
	# print (trn_embedding, val_embedding)
	# print (triplet_model.get_weights())

	clf = SVC(
		# class_weight='balanced',
		probability=True,
		# tol=1e-4,
	)
	

	clf.fit(trn_embedding, Y_trn)

	print (clf.score(val_embedding, Y_val))
	print (clf.predict_proba(val_embedding))

	print (roc_auc_score(Y_val, clf.predict(val_embedding)))
	print (classification_report(Y_val, clf.predict(val_embedding), digits=4))


	all_files = [x[:-8] for x in os.listdir(ALL_FILES)]
	X = [pickle.load(open(os.path.join(FEATURE_PATH, x + ".fkmeans"), "rb"), encoding='latin1') for x in all_files]
	# Y = [ranks[x.split()[0].strip()] for x in all_files]

	proba = clf.predict_proba(embed(np.array(X), triplet_model))

	# print(len(proba), len(proba[0]), proba[0])

	wf = open("proba_audio_wo_pretrain.txt", "w")
	for i, [_, prob] in enumerate(proba):
		wf.write("%s,%.5f\n" % (all_files[i], prob))





	


