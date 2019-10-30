import os, sys, numpy as np, pickle, time
from sklearn.cluster import MiniBatchKMeans


FEATURE_PATH = "./audio_features/"
KMEANS_PATH = "./mfcc_kmeans_200/"
N_CLUSTERS = 200

if not os.path.isdir(KMEANS_PATH):
	os.mkdir(KMEANS_PATH)

all_mfcc, record = [], []
# ld_set = set(os.listdir(FEATURE_PATH))



# s = open("s3_file.lst")
# lines = s.readlines()
# ld = [x.split("/")[1][:-5] + ".feature" for x in lines]

# print (len(ld_set))

# print ("1herocs - 2019-03-12 21h46m08s - Aloha Hero  follow me on twitter zuricarter.feature" in ld)

# print (ld[0])
# print (list(ld_set)[0])

# assert all (x in ld_set for x in ld)

# for x in ld:
# 	if x not in ld_set:
# 		print (x)

# assert False

ld = os.listdir(FEATURE_PATH)

ld = [x for x in ld if x.endswith(".feature")]
assert all(x.endswith(".feature") for x in ld)


def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]


for fname in ld:
	feature = pickle.load(open(os.path.join(FEATURE_PATH, fname), "rb"))
	mfcc = feature["mfcc"].T

	if mfcc.shape[1] != 20:
		print ("Exception:", fname)
		continue

	record.append(mfcc)
	for frame in mfcc:
		all_mfcc.append(frame)


print ("Total number of frames:", len(all_mfcc))
print ("Read MFCC feature completed")

sys.stdout.flush()

if len(sys.argv) == 1:
	kmeans = MiniBatchKMeans(
		n_clusters=N_CLUSTERS,
		batch_size=N_CLUSTERS*200,
		max_no_improvement=20,
		reassignment_ratio=1e-3,
		verbose=1,
	)
	kmeans.fit(all_mfcc)
	pickle.dump(kmeans, open("mfcc_kmeans.pickle", "wb"))

else:
	kmeans = pickle.load(open(sys.argv[1], "rb"))

centroids = kmeans.cluster_centers_
print ("Mini Batch K-Means clustering done.")
pickle.dump(kmeans, open("kmeans_200", "wb"))

sys.stdout.flush()


for i, fname in enumerate(ld):

	if not i % 100:
		print (i, "/", len(ld))

	wf = os.path.join(KMEANS_PATH, fname[:-6] + "kmeans")
	vec = np.zeros(N_CLUSTERS)
	for frame in record[i]:
		dist = [np.linalg.norm(c - frame) for c in centroids]
		vec[argmin(dist)] += 1.

	with open(wf, "wb") as handle:
		pickle.dump(vec, handle, protocol=2)

	sys.stdout.flush()

print ("K-Means feature generated successfully!")


