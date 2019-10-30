import os, sys, pickle, numpy as np, json
from sklearn.svm.classes import SVC
import keras
import keras.backend as K

FEATURE = "./mfcc_representation_50"
# FEATURE = "./mfcc_kmeans_esea"

if __name__ == '__main__':

	ranks = json.loads(open("player_rank.json", "r").readline())
	X, Y = [], []
	X_val, Y_val = [], []

	trn_f = open("trn_esea.lst", "r")
	trn = [x.strip("\n") for x in trn_f.readlines()]
	trn_f.close()

	val_f = open("val_esea.lst", "r")
	val = [x.strip("\n") for x in val_f.readlines()]
	val_f.close()

	for fname in trn:
		X.append(pickle.load(open(os.path.join(FEATURE, fname+".50dim"), "rb")))
		Y.append(ranks[fname.split()[0].strip()])

	for fname in val:
		X_val.append(pickle.load(open(os.path.join(FEATURE, fname+".50dim"), "rb")))
		Y_val.append(ranks[fname.split()[0].strip()])

	cnt = {1: 695, 0: 805}
	class_weight = {x: 1./y for x, y in cnt.items()}

	X = np.array(X)
	Y = np.eye(2)[Y]
	X_val = np.array(X_val)
	Y_val = np.eye(2)[Y_val]

	model = keras.Sequential()
	model.add(keras.layers.Dense(units=25, input_shape=(50,), activation='relu', name='dense_1'))
	model.add(keras.layers.Dropout(0.5, name='dropout_1'))
	# model.add(keras.layers.Dense(units=10, activation='relu', name='dense_2'))
	# model.add(keras.layers.Dropout(0.5, name='dropout_2'))
	model.add(keras.layers.Dense(units=2, activation='softmax', name='output'))
	model.summary()

	adam_optimizer = keras.optimizers.Adam(lr=1e-4)
	model.compile(
		loss='binary_crossentropy',
		optimizer=adam_optimizer,
		metrics=['accuracy'],
	)

	checkpoint = keras.callbacks.ModelCheckpoint(
		filepath='mfcc_esea.best.hdf5',
		monitor='val_acc',
		verbose=1,
		save_best_only=True,
		mode='auto',
		period=50,
	)

	early_stopping = keras.callbacks.EarlyStopping(
		monitor='val_acc',
		min_delta=0,
		patience=100,
		verbose=1,
		mode='auto',
	)

	model.fit(
		X,
		Y,
		epochs=50000,
		batch_size=500,
		validation_data=(X_val, Y_val),
		class_weight=class_weight,
		validation_freq=50,
		verbose=1,
		callbacks=[checkpoint, early_stopping],
	)

	# clf = SVC(
	# 	gamma="auto",
	# 	probability=True,
	# 	decision_function_shape="ovr",
	# 	class_weight="balanced",
	# )
	# clf.fit(X, Y)
	# pickle.dump(clf, open("mfcc_SVM.model", "wb"))

	# print (clf.score(X_val, Y_val))
