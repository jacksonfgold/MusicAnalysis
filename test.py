import pycurl

with open("test.mp3", 'wb') as f:
	c = pycurl.Curl()
	c.setopt(c.URL, 'http://e-cdn-preview-8.deezer.com/stream/8e564ba9cde487278546b33c12ee1266-1.mp3')
	c.setopt(c.WRITEDATA, f)
	c.perform()
	c.close()