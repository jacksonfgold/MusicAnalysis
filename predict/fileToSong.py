from createSong import createSong
import os

directory = "predict/songdatabase"

for song in os.listdir(directory):
	createSong(directory +"/" + song, song[:-4])