import os, subprocess

"""
ffmpeg -y -i videofile -ac 1 -f wav audiofile

"""

VIDEO = "./video/"
AUDIO = "./audio/"
S3_PATH = "s3://11775projecttwitchvideostore/allusers-wav/"

for video in os.listdir(VIDEO):
	videofile = os.path.join(VIDEO, video)
	audiofile = os.path.join(AUDIO, video[:-3]+"wav")
	subprocess.call([
		'ffmpeg',
		'-y',
		'-i',
		videofile,
		'-ac',
		'1',
		'-f',
		'wav',
		audiofile,
	])

	print ("%s extracted, moving to s3" % audiofile)

	subprocess.call([
		'aws',
		's3',
		'mv',
		audiofile,
		S3_PATH,
	])


