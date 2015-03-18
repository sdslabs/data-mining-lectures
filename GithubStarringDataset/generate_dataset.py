from random import randrange

lang = ['python', 'cpp', 'html', 'css', 'php']
org = ['SDSLabs', 'PAG', 'lisa-lab', 'dolores']

for i in range(50):
	idx = randrange(0, len(lang))
	print "'" + lang[idx] + "',",
	idx = randrange(0, len(org))
	print "'" + org[idx] + "',",
	idx = randrange(0, 2)
	print str(idx) + ',',
	idx = randrange(0, 100)
	print str(idx)