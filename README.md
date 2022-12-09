# Music Analysis Model

This model analyzes given songs by mapping them to the latent space created by the Word2Vec word embeddings.

## Downloading

Download the [Wasabi Dataset](https://github.com/micbuffa/WasabiDataset) and put the JSON files in a folder called "wasabi". 

Run getSongs.py to download preview of 4000 songs that will be put in the songs directory. **Warning:** This is quite memory intensive

If you are on Windows, as torchaudio does not support mp3 files, run toWav.py to convert files to wav.

You can run getSongInfo.py with a specfic song_id to see the emotion or social tags of the given song_id.

Download the [Google News Word2Vec weights](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g) and put them in the weights folder.

If you want to use pre-trained version of the model, download weights [here](https://drive.google.com/file/d/1FCm4H0NcvD6mo2sA5DndPyBl9h0atdHP/view?usp=sharing) and place inside "weights" folder

## Training

### Pre-Proccessing 

Run the data.py to preprocess data and serialize to file. Set input directory to correct one, songs if mp3, songs2 if wav. 

### Train

In train.py, set epochs to desired amount and run. Note that the loss function is quite time complex so training can be abnormally long per batch.

## Predictions

To find the most similar song from a set of songs, first place all songs in the "songdatabase" folder within the "predict" folder. To get the list of word embeddings for each song in the file, run "fileToSong.py" which will then serialize a list of all of the embeddings to file. 

Next, in "findSong.py" put the input words of your choice in the "words" list and run the file. This should return the most similar song from the folder of songs to the input.



