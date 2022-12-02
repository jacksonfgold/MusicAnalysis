import json

f = open("wasabi/social-tags.json")

emotions = json.load(f)

for song in emotions:
	#print(song)
	if song["song_id"]["$oid"] == "5714dec325ac0d8aee383fa1":
		print(song)
		break