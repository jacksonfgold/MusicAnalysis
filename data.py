import os
import torchaudio

directory = "songs"

for song in os.listdir(directory)[:50]:
	print(song)
