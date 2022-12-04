import os
from pydub import AudioSegment

for song in os.listdir("songs"):
	input_file = "songs/" + song 
	output_file = "songs2/" + song[:-4] + ".wav"
	try:
		sound = AudioSegment.from_mp3(input_file)
	except:
		continue
	sound.export(output_file, format="wav")