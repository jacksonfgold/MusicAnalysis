import os
import torchaudio
import warnings
import numpy as np
import pickle

warnings.filterwarnings(action = 'ignore')
 
import gensim
from gensim.models import Word2Vec

directory = "songs"

import json

f = open("wasabi/social-tags.json")
g = open("wasabi/emotion-tags.json")

social = json.load(f)
emotions = json.load(g)

def retrieveSocialTags(song_id):
	for song in social:
		if song["song_id"]["$oid"] == song_id:
			return [x["social_tag"].replace(" ", "-") for x in song["socials"]]

def retrieveEmotionTags(song_id):
	print(song_id)

	for song in emotions:
		if song["song_id"]["$oid"] == song_id:
			return [x["emotion_tag"] for x in song["emotions"]]


word2vec = gensim.models.KeyedVectors.load_word2vec_format('weights/GoogleNews-vectors-negative300.bin', binary=True)  

def convertWords(words):

	word_vectors = []

	word_count = 0

	for word in words:
		if word in word2vec:
			vec = word2vec[word]
			word_vectors.append(vec)
			word_count+=1;
		if word_count == 50:
			break

	while word_count < 50:
		word_vectors.append(np.zeros(300))
		word_count += 1


	return word_vectors


song_tensors = []
song_labels = []

for song in os.listdir(directory):

	try:
		audio, w = torchaudio.load("songs/" + song, format="mp3")
	except:
		continue

	emot = retrieveEmotionTags(song[:-4])

	soc = retrieveSocialTags(song[:-4])

	if (not emot or not soc):
		continue

	tags = convertWords(emot + soc)[:50]

	song_labels.append(tags)
	song_tensors.append(audio)

with open('data/song_tensors.pkl', 'wb') as f:
	pickle.dump(song_tensors, f)

with open('data/song_labels.pkl', 'wb') as f:
	pickle.dump(song_labels, f)


f.close()
g.close()
	