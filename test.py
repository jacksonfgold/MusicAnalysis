import gensim

word2vec = gensim.models.KeyedVectors.load_word2vec_format('weights/GoogleNews-vectors-negative300.bin', binary=True)  


words = ['beautiful', 'ugly']

print(model['beautiful'])

