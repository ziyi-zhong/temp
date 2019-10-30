import numpy as np
import pickle, librosa, os, time, sys

INPUT = "./audio/"
OUTPUT = "./audio_features_esea/"
SR = 8000

if __name__ == '__main__':
	for file_name in os.listdir(INPUT):

		features = {}
		try:
			start_time = time.time()
			audio_file = os.path.join(INPUT, file_name)
			output_file = os.path.join(OUTPUT, file_name[:-3] + "feature")

			y, sr = librosa.load(audio_file, sr=SR)
			S = np.abs(librosa.stft(y))
			features["mfcc"] = librosa.feature.mfcc(y=y, sr=sr)
			features["rms"] = librosa.feature.rms(y=y)
			features["rmse"] = librosa.feature.rmse(y=y)
			features["stft"] = librosa.feature.chroma_stft(y=y, sr=sr)
			features["flatness"] = librosa.feature.spectral_flatness(y=y)
			features["contrast"] = librosa.feature.spectral_contrast(S=S, sr=sr, fmin=100)
			with open(output_file, "wb") as handle:
				pickle.dump(features, handle, protocol=2)
			print (file_name, "dump done. --- %s seconds ---" % (time.time() - start_time))

		except Exception as e:
			with open(output_file, "wb") as handle:
				pickle.dump(features, handle, protocol=2)
			print (file_name, "Exception happened.  --- %s seconds ---" % (time.time() - start_time))

		sys.stdout.flush()




