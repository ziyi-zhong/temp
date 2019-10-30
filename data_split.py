import os, sys, random, json
from collections import defaultdict

PATH = "./mfcc_kmeans_500/"
all_files = [x[:-8] for x in os.listdir(PATH)]
assert all(os.path.isfile(os.path.join(PATH, f + ".fkmeans")) for f in all_files)

player_info = json.load(open("num_of_followers.json"))

random.shuffle(all_files)

trn, val = open("trn.lst", "w"), open("val.lst", "w")
# for i, f in enumerate(all_files):
# 	if i < 1500:
# 		trn.write("%s\n" % f)
# 	else:
# 		val.write("%s\n" % f)

# trn.close()
# val.close()

### stratification

thres = 5007. / 7007

d = defaultdict(list)

for fname in all_files:
	user = fname.split()[0].strip()
	# print (user)

	d[player_info[user]["cluster"]].append(fname)

for _, files in d.items():
	l = len(files)
	for i, f in enumerate(files):
		if float(i) / l < thres:
			trn.write("%s\n" % f)
		else:
			val.write("%s\n" % f)

trn.close()
val.close()

