import keras
import json, numpy as np, pickle, sys, os
from keras.regularizers import l1, l2, l1_l2
import keras.backend as K


FEATURE_PATH = "./audio_features/"


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
		feat_file = os.path.join(FEATURE_PATH, fname+".feature")
		feat = fix_size(pickle.load(open(feat_file, "rb"))["mfcc"])
		X_trn.append(feat)
		Y_trn.append(player_info[user]['cluster'])

	for fname in val_files:
		user = fname.split()[0].strip()
		feat_file = os.path.join(FEATURE_PATH, fname+".feature")
		feat = fix_size(pickle.load(open(feat_file, "rb"))["mfcc"])
		X_val.append(feat)
		Y_val.append(player_info[user]['cluster'])

	X_trn = np.array(X_trn)
	Y_trn = np.eye(10)[Y_trn]
	X_val = np.array(X_val)
	Y_val = np.eye(10)[Y_val]

	if K.image_data_format() == 'channels_first':
		X_trn = X_trn.reshape(X_trn.shape[0], 1, 20, 4688)
		X_val = X_val.reshape(X_val.shape[0], 1, 20, 4688)
		input_shape = (1, 20, 4688)
	else:
		X_trn = X_trn.reshape(X_trn.shape[0], 20, 4688, 1)
		X_val = X_val.reshape(X_val.shape[0], 20, 4688, 1)
		input_shape = (20, 4688, 1)


	model = keras.Sequential()
	model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1),
					 activation='relu',
					 input_shape=input_shape))
	# model.add(keras.layers.Dropout(0.1))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	# model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
	# model.add(keras.layers.Dropout(0.2))
	# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Flatten(name="flatten"))
	model.add(keras.layers.Dense(50, activation='relu', name="dense1"))
	# model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.Dense(50, activation='relu', name="dense2"))
	model.add(keras.layers.Dropout(0.1))
	model.add(keras.layers.Dense(10, activation='softmax'))

	model.summary()

	# adam_optimizer = keras.optimizers.Adam(lr=5e-3, decay=1e-6)
	model.compile(
		loss='categorical_crossentropy',
		# optimizer=adam_optimizer,
		optimizer='adam',
		metrics=['accuracy'],
	)

	trn_class_cnt = {4: 987, 0: 891, 5: 894, 7: 236, 6: 93, 9: 805, 2: 535, 3: 470, 8: 335, 1: 156}
	class_weight = {x: 1./y for x, y in trn_class_cnt.items()}

	checkpoint = keras.callbacks.ModelCheckpoint(
		filepath='mfcc_pretrain_weights_conv.best.hdf5',
		monitor='val_acc',
		verbose=1,
		save_best_only=True,
		mode='auto',
		period=1,
	)

	early_stopping = keras.callbacks.EarlyStopping(
		monitor='trn_acc',
		min_delta=0,
		patience=100,
		verbose=1,
		mode='auto',
	)

	model.fit(
		X_trn,
		Y_trn,
		epochs=10000,
		# batch_size=500,
		validation_data=(X_val, Y_val),
		class_weight=class_weight,
		validation_freq=1,
		verbose=1,
		callbacks=[checkpoint, early_stopping],
	)

