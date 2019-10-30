import numpy as np
import json
import keras
import keras.backend as K
import tensorflow as tf
import os
import pickle



FEATURE_PATH = "./audio_features_esea/"

def fix_size(mat, s=4688):
	x, y = mat.shape
	assert x == 20
	if y == s:
		return mat

	elif y > s:
		return mat[:, :s]

	else:
		temp = np.zeros((20, s))
		temp[:x, :y] = mat
		return temp

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
		user = fname.split()[0].strip()
		feat_file = os.path.join(FEATURE_PATH, fname+".feature")
		feat = fix_size(pickle.load(open(feat_file, "rb"))["mfcc"])
		X.append(feat)
		Y.append(ranks[fname.split()[0].strip()])

	for fname in val:
		user = fname.split()[0].strip()
		feat_file = os.path.join(FEATURE_PATH, fname+".feature")
		feat = fix_size(pickle.load(open(feat_file, "rb"))["mfcc"])
		X_val.append(feat)
		Y_val.append(ranks[fname.split()[0].strip()])


	X = np.array(X)
	Y = np.eye(2)[Y]
	X_val = np.array(X_val)
	Y_val = np.eye(2)[Y_val]

	if K.image_data_format() == 'channels_first':
		X = X.reshape(X.shape[0], 1, 20, 4688)
		X_val = X_val.reshape(X_val.shape[0], 1, 20, 4688)
		input_shape = (1, 20, 4688)
	else:
		X = X.reshape(X.shape[0], 20, 4688, 1)
		X_val = X_val.reshape(X_val.shape[0], 20, 4688, 1)
		input_shape = (20, 4688, 1)

	# define model
	embedding_dim = 20
	filter_sizes = [2, 3, 4]
	num_filters = 128

	inputs = keras.layers.Input(shape=input_shape, dtype='float32')
	conv_0 = keras.layers.Conv2D(
		filters=num_filters,
		kernel_size=(filter_sizes[0], embedding_dim),
		padding='valid',
		activation='relu',
		kernel_initializer='normal',
	)(inputs)

	conv_1 = keras.layers.Conv2D(
		filters=num_filters,
		kernel_size=(filter_sizes[1], embedding_dim),
		padding='valid',
		activation='relu',
		kernel_initializer='normal',
	)(inputs)
	conv_2 = keras.layers.Conv2D(
		filters=num_filters,
		kernel_size=(filter_sizes[2], embedding_dim),
		padding='valid',
		activation='relu',
		kernel_initializer='normal',
	)(inputs)

	maxpool_0 = keras.layers.GlobalMaxPool2D()(conv_0)
	maxpool_1 = keras.layers.GlobalMaxPool2D()(conv_1)
	maxpool_2 = keras.layers.GlobalMaxPool2D()(conv_2)

	concat_tensor = keras.layers.Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
	hidden = keras.layers.Dense(units=10, activation='relu')(concat_tensor)
	output = keras.layers.Dense(units=2, activation='softmax')(hidden)

	model = keras.models.Model(inputs=inputs, outputs=output)

	model.summary(line_length=100)
	checkpoint = keras.callbacks.ModelCheckpoint(
		filepath='weights_cnn_ng.best.hdf5',
		monitor='val_acc',
		verbose=1,
		save_best_only=True,
		mode='auto',
		period=1,
	)

	early_stopping = keras.callbacks.EarlyStopping(
		monitor='val_acc',
		min_delta=0,
		patience=10,
		verbose=0,
		mode='auto',
	)

	model.compile(
		loss='binary_crossentropy',
		optimizer='adam',
		metrics=['accuracy']
	)

	cnt = {1: 695, 0: 805}
	class_weight = {x: 1./y for x, y in cnt.items()}

	model.fit(
		X,
		Y,
		epochs=10000,
		# batch_size=500,
		validation_data=(X_val, Y_val),
		class_weight=class_weight,
		validation_freq=1,
		verbose=1,
		callbacks=[checkpoint, early_stopping],
	)

