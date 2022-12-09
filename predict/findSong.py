import os
import torchaudio
import warnings
import numpy as np
import pickle
import torch

#warnings.filterwarnings(action = 'ignore')

words = ['cheese', 'guitar', 'rock', 'vocals']
 
import gensim
from gensim.models import Word2Vec

word2vec = gensim.models.KeyedVectors.load_word2vec_format('weights/GoogleNews-vectors-negative300.bin', binary=True)  

max_words = 20

def convertWords(words):

	word_vectors = []

	word_count = 0

	for word in words:
		if word in word2vec:
			vec = word2vec[word]
			word_vectors.append(torch.tensor(vec))
			word_count+=1;
		if word_count == max_words:
			break

	while word_count < max_words:
		word_vectors.append(torch.tensor(np.zeros(300)))
		word_count += 1

	return word_vectors

def findSong(words):

	with open('songs.pkl', 'rb') as f:
	    x = pickle.load(f)

	word_vecs = convertWords(words)
	print(word_vecs)
	min_dist = 10000000
	min_song = ""

	for song in x:
		min_dist = 10000000
		min_song = ""
		dist = song.distance(word_vecs)
		if (dist<min_dist):
			min_dist = dist
			min_song = song.name

	return min_song

print(findSong(words))


