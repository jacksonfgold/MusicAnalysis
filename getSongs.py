import pycurl
import json

f = open("wasabi/song.json",  encoding="utf8")

songs = json.load(f)

i = 0

for song in songs:

	# If there is a preview of the song, download it
	preview = song["preview"]
	if preview and song["has_emotion_tags"] == "True" and song["has_social_tags"] == "True":
		print(preview, song["has_emotion_tags"])
		file_name = "songs/" + song["_id"]["$oid"] + ".mp3"
		with open(file_name, 'wb') as f:
		    c = pycurl.Curl()
		    c.setopt(c.URL, preview)
		    c.setopt(c.WRITEDATA, f)
		    c.perform()
		    c.close()
		i+=1
	if i > 25000:
		break


