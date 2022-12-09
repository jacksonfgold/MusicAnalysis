from song import Song
import pickle
import os

path= "C:/Users/jacks/egg/MusicAnalysis/predict/songdatabase/Marigold.wav"

name = "Marigold"

def createSong(path, name):
	x = []

	if os.path.exists("songs.pkl") > 0:
		with open('songs.pkl', 'rb') as f:
			x = pickle.load(f)

	song = Song(path, name)

	x.append(song)

	with open('songs.pkl', 'wb') as f:
	    pickle.dump(x, f)


