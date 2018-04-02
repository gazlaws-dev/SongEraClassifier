from PyLyrics import *
import pandas as pd
import sys



listOfDict = []

df = pd.read_csv('billboard_lyrics_1964-2015.csv')

for index, row in df.iterrows():
	print(row["Song"])
	print(row["Artist"])
	print(index)
	try:
		lyric = PyLyrics.getLyrics(row["Song"],row["Artist"])
		newEntry = {"Year":row["Year"], "Lyrics":lyric}
		listOfDict.append(newEntry)
	except:
		continue
	

df = pd.DataFrame(listOfDict)
df.to_csv('mytestop.csv')

print("Done making csv!")

