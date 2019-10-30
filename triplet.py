import keras
import keras.backend as K
import numpy as np, json, pickle, time, sys, os, codecs

FEATURE_PATH = "./mfcc_kmeans_200/"

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
	Using Square Euclidean distance and maxplus margin.
	"""
	anchor, pos, neg = inputs
	pos_dist = K.square(anchor - pos)
	neg_dist = K.square(anchor - neg)

	# Square Euclidean Distance
	pos_dist = K.sum(pos_dist, axis=-1, keepdims=True)
	neg_dist = K.sum(neg_dist, axis=-1, keepdims=True)

	# Maxplus Margin
	loss = pos_dist - neg_dist
	# return K.mean(K.log(1 + K.exp(loss)))
	return K.mean(K.maximum(0.0, 1. + loss))


def triplet_generator(files, ranks, batch_size=100):

	f = open(files, "r")
	file_lst = [x.strip('\n') for x in f.readlines()]
	f.close()

	# assert all(os.path.isfile(os.path.join(FEATURE_PATH, x)) for x in file_lst)

	X = [pickle.load(open(os.path.join(FEATURE_PATH, x + ".fkmeans"), "rb"), encoding='latin1') for x in file_lst]
	Y = [ranks[x.split()[0].strip()] for x in file_lst]

	X = list(map(normalize, X))

	all_pos = [X[i] for i, y in enumerate(Y) if y == 1]
	neg = [X[i] for i, y in enumerate(Y) if y == 0]

	l = len(all_pos)
	pivot = l // 2 if l % 2 else (l // 2 + 1)
	anchor, pos = all_pos[:pivot], all_pos[pivot:]

	ind_anchor, ind_pos, ind_neg = 0, 0, 0

	# Generate
	while True:

		anchor_lst, pos_lst, neg_lst = [], [], []
		for _ in range(batch_size):
			# anchor_lst.append(normalize(anchor[ind_anchor]))
			# pos_lst.append(normalize(pos[ind_pos]))
			# neg_lst.append(normalize(neg[ind_neg]))
			anchor_lst.append(anchor[ind_anchor])
			pos_lst.append(pos[ind_pos])
			neg_lst.append(neg[ind_neg])
			ind_anchor = (ind_anchor + 1) % len(anchor)
			ind_pos = (ind_pos + 1) % len(pos)
			ind_neg = (ind_neg + 1) % len(neg)

		A = np.array(anchor_lst)
		P = np.array(pos_lst)
		N = np.array(neg_lst)
		# yield ([[anchor_lst[i], pos_lst[i], neg_lst[i]] for i in range(batch_size)], None)
		yield ({"anchor_input": A, "pos_input": P, "neg_input": N}, None)


def get_mlp(input_shape=(200,), pretrain_model=None):
	mlp = keras.Sequential()
	mlp.add(keras.layers.Dense(units=100, input_shape=input_shape, activation='sigmoid', name='dense_1'))
	mlp.add(keras.layers.Dropout(0.1, name='dropout_1'))
	mlp.add(keras.layers.Dense(units=50, activation='sigmoid', name='dense_2'))

	if pretrain_model is not None:
		mlp.add(keras.layers.Dropout(0.1, name='dropout_2'))
		mlp.add(keras.layers.Dense(units=10, activation='softmax', name='output'))
		mlp.load_weights(pretrain_model)
		mlp.pop()
		mlp.pop()

	return mlp

def get_model(input_shape=(200,), pretrain_model=None):
	mlp = get_mlp(input_shape, pretrain_model)
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


def test():
	f = open("val_esea_real.lst", "r")
	file_lst = [x.strip('\n') for x in f.readlines()]
	f.close()

	assert all(os.path.isfile(os.path.join(FEATURE_PATH, x + ".fkmeans")) for x in file_lst)

if __name__ == '__main__':

	test()


	mlp, triplet_model = get_model(pretrain_model='mfcc_pretrain_weights_real.best.hdf5')
	ranks = get_player_rank()
	gen_trn = triplet_generator("trn_esea_real.lst", ranks)
	gen_val = triplet_generator("val_esea_real.lst", ranks)

	for layer in mlp.layers[:-1]:
		layer.trainable = True
	mlp.layers[-1].trainable=False

	triplet_model.summary()
	triplet_model.compile(loss=None, optimizer=keras.optimizers.Adam())

	triplet_model.fit_generator(
		generator=gen_trn,
		validation_data=gen_val,
		verbose=1,
		steps_per_epoch=500,
		epochs=60,
		# callbacks=[checkpoint],
		validation_steps=10,
	)

	triplet_model.save_weights("triplet_pre_60.best.hdf5")
	mlp.save_weights("triplet_mlp_pre_60.best.hdf5")



