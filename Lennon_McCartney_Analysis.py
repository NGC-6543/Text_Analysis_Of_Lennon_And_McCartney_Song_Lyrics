# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:03:43 2020

@author: NGC-6543




"""

####################################################
## Part 1 Lyrics from API - This part requires a key to run 
####################################################

# Part 1 retrieves the song lyrics from the API and prints them to a file.
# An separate API key is required to run this part.  
# However, this part may be skipped since the files generated in this part
# will be provided separately.

# a key to canarado-lyrics.p.rapidapi.com was obtained from rapidapi.com

# Files required to run part 1:
# beatles_song_list_utf8.txt


#%% Packages

import requests
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os

#%% Set working directory

os.getcwd()
#os.chdir()
#os.getcwd()


#%% Read in list of beatles songs with attributions

songsDF = pd.read_csv('beatles_song_list_utf8.txt',sep='\t',encoding='utf-8')



#%% Test the API (this can be skipped if you know it works.)
######################################################

# Hey,%2520Hey,%2520Hey,%2520Hey%2520the%2520beatles
# Blackbird%2520the%2520beatles

try:
    url = "https://canarado-lyrics.p.rapidapi.com/lyrics/Blackbird%2520the%2520beatles"
    
    headers = {
        'x-rapidapi-host': "canarado-lyrics.p.rapidapi.com",
        'x-rapidapi-key': "your key here"
        }
    
    response = requests.request("GET", url, headers=headers)
    print('GET succeeded')
    
except:
    print('GET caused error')

print(response.text)


print(response.url)
# check encoding of response
response.encoding
'utf-8'
# look at response content
response.content
# decode content as json to see if json looks right
response.json()
# get response data type
type(response)

# Now use json library to convert to a python dictionary
jsontxt = response.json()
print(jsontxt)
type(jsontxt)

jsontxt.keys() #dict_keys(['status', 'content'])
jsontxt['content'][0].keys() # dict_keys(['title', 'lyrics', 'artist'])

jsontxt['content'][0]['artist']
jsontxt['content'][0]['title']
jsontxt['content'][0]['lyrics']


#%% Prepare each song title for each API query
######################################################
html_sep = '%2520'
suffix = '%2520The%2520Beatles'

song_strings = []

for song in songsDF.Song:
    song_split = song.split()
    song_string = html_sep.join(song_split)
    song_string = song_string + suffix
    song_strings.append(song_string)
    
del html_sep, suffix, song, song_split, song_string   

songs = list(songsDF.Song)
songs = list(zip(songs,song_strings))
song_stringsDF = pd.DataFrame(songs,columns=['Song','Song_string'])

del songs, song_strings



#%% Definition to run all the queries and return them as dataframes
######################################################

def get_all_lyrics(song, song_string):

    
    url = "https://canarado-lyrics.p.rapidapi.com/lyrics/" + song_string
    
    try:
        
        headers = {
            'x-rapidapi-host': "canarado-lyrics.p.rapidapi.com",
            'x-rapidapi-key': "your key here"
            }
        
        response = requests.request("GET", url, headers=headers)
        print('GET succeeded')
        
    except:
        print('GET caused an error')
    
    jsontxt = response.json()
    
    jsontxt['content'][0]
    
    songs = []
    response = []
    artists = []
    titles = []
    lyrics = []
    
    i = 0
    for info in jsontxt['content']:
        
        artist = info['artist']
        title = info['title']
        lyric = info['lyrics']
        
        songs.append(song)
        response.append(i)
        artists.append(artist)
        titles.append(title)
        lyrics.append(lyric)
        
        i+=1
        
     
    songs_dict = {'song' : songs
                  ,'response' : response
                  ,'artist' : artists
                  ,'title' : titles
                  ,'lyrics' : lyrics}
    
    new_songsDF = pd.DataFrame(songs_dict)
    
    return new_songsDF



#%% Run all the queries and contatenate the results of each one into a dataframe
######################################################
    
all_track_info = pd.DataFrame()
for row in song_stringsDF.itertuples():
    print(row.Song, row.Song_string)
    new_songsDF = get_all_lyrics(row.Song, row.Song_string)
    all_track_info = pd.concat([all_track_info,new_songsDF])



#%% Keep only the records in the results that are actual Beatles songs 
## (It's always the first reference from each query)
##############################################################################


all_track_info.loc[all_track_info['response']==0]
all_track_info_short = all_track_info.loc[all_track_info['response']==0]
all_track_info_short2 = all_track_info_short.loc[: , ['song','lyrics']]
all_track_info_short2 = all_track_info_short2.rename(columns={"song":"Song","lyrics":"Lyrics"})

# It's the only song the API had no correct result for.
# Removed it from the input file so this is not necessary
# songsDF.loc[songsDF['Song']=='If I Need Someone '] #78
# songsDF = songsDF[songsDF.Song != 'If I Need Someone']

#Merge the dataframes
songsDF_lyrics = pd.merge(songsDF,all_track_info_short2,how='left',on='Song')


#%% Remove special characters
######################################################

# we will need to remove any [ (any content between them) ] 
# and special characters like \n from lyrics that have them

def replaceString(string):
    
    string = string.replace('\n',' ')
    string = string.replace('\r',' ')
    string = string.replace('\t',' ')
    
    return string

songsDF_lyrics['Lyrics'] = songsDF_lyrics['Lyrics'].apply(lambda x: replaceString(x))


def replaceBrackets(string):
    
    string = re.sub(r'\[.*?\]', ' ', string, flags=re.IGNORECASE)
    string = re.sub(r'\{.*?\}', ' ', string, flags=re.IGNORECASE)

    return string

songsDF_lyrics['Lyrics'] = songsDF_lyrics['Lyrics'].apply(lambda x: replaceBrackets(x))




#%% Output the dataframes to tab-delimited text files
######################################################

songsDF_lyrics.to_csv('songsDF_lyrics_utf8.txt',sep='\t', index = False, encoding='utf-8')



####################################################
## Part 2 Process text files
####################################################

# In this part, the files are cleaned up and prepared for downstream analysis.
# Additionally, VADER sentiment scores are generated for each song's lyrics.

# Files required to run part 2:
# songsDF_lyrics_utf8.txt (from part 1)
# PM_CR.csv
# JL_CR.csv



import pandas as pd
import numpy as np
import os



#%% strip whitespace from files


pm_file2 = open('PM2.csv', 'w')
pm_file = open('PM_CR.csv', 'r')
pm_file2.write('Song,Lyrics\n')
for line in pm_file:
    line = line.lstrip(' ')
    pm_file2.write(line)
pm_file.close()
pm_file2.close()

jl_file2 = open('JL2.csv', 'w')
jl_file = open('JL_CR.csv', 'r')
jl_file2.write('Song,Lyrics\n')
for line in jl_file:
    line = line.lstrip(' ')
    jl_file2.write(line)
jl_file.close()
jl_file2.close()


#%% load McCartney solo songs

pm_songs = pd.read_csv('PM2.csv'
                       , sep=','
                       , encoding='utf-8'
                       #, lineterminator = '\n'
                       , quotechar = '"')

pm_songs.insert(0, 'Composer', 'M', allow_duplicates = False)

# pm_songs = pm_songs.rename(columns = {'Songs':'Songs','Lyrics\r':'Lyrics'})

#%% load Lennon solo songs

jl_songs = pd.read_csv('JL2.csv'
                       , sep=','
                       , encoding='utf-8'
                       #, lineterminator = '\n'
                       , quotechar = '"')

jl_songs.insert(0, 'Composer', 'L', allow_duplicates = False)

#%%

beatles_songs = pd.read_csv('songsDF_lyrics_utf8.txt'
                            , sep='\t'
                            , encoding='utf-8')

beatles_songs['Song'] = beatles_songs['Song'].map(lambda x: x.replace("´", "'"))



#%%  Fuzzy match song titles vs. Beatles songs

# Eliminate song lyrics duplicates by fuzzy matching on the song titles
# This is necessary because there are duplicates in the Beatles, Lennon, and 
# McCartney files used to start the analysis.


from fuzzywuzzy import fuzz

def compare_song_titles(song1, song2):
    
    Str1 = song1
    Str2 = song2
    
    Ratio = fuzz.ratio(Str1.lower(),Str2.lower())

    return Ratio


#%% Duplicates in Lennon songs and Beatles songs
    
for row1 in jl_songs.itertuples():
    for row2 in beatles_songs.itertuples():
        ratio = compare_song_titles(row1.Song, row2.Song)
        if ratio >= 70:
            print(row1.Song, '\t', row2.Song, '\t', ratio)
            print(row1.Lyrics[0:80])
            print(row2.Lyrics[0:80], '\n')
    
#%% Clean-up Lennon songs
    

JL_duplicates = [
'Come Together'
,'Maggie Mae'
,'I Saw Her Standing There'
,'Dizzy Miss Lizzy'
,'Honey Don\'t'
,'Love Me Do'
,'Lucy In The Sky With Diamonds'
,'The Ballad Of John And Yoko'
,'You Won\'t See Me'
]

jl_songs = jl_songs.loc[~jl_songs['Song'].isin(JL_duplicates)]
jl_songs.reset_index(inplace = True, drop = True)

#%% Clean-up Lennon songs (2)
    

jl_internal_dups = [136,77,140,141,39,89,144,145,143,146,83,17,150,149,24,153,82,91,92,158,159,93,162,85,26,56,98,99]
jl_songs = jl_songs.loc[~jl_songs.index.isin(jl_internal_dups)]
jl_songs.reset_index(inplace = True, drop = True)


#%% Duplicates in McCartney songs and Beatles songs

duplicates = []
for row1 in pm_songs.itertuples():
    for row2 in beatles_songs.itertuples():
        ratio = compare_song_titles(row1.Song, row2.Song)
        if ratio >= 70:
            match =  str(row1.Song) + '\t' + str(row2.Song) + '\t' + str(ratio) + '\n'
            
            duplicates.append(match)


#%% Clean-up McCartney songs

pm_dup_songs = [
"All My Loving"
,"And I Love Her"
,"Back In The U.S.S. R"
,"Back In The U.S.S.R."
,"Birthday"
,"Blackbird"
,"Can't Buy Me Love"
,"Carry That Weight"
,"Day Tripper"
,"Drive My Car"
,"Eleanor Rigby"
,"For No One"
,"For You Blue"
,"Get Back"
,"Getting Better"
,"Girl"
,"Got To Get You Into My Life"
,"Hello Goodbye"
,"Help!"
,"Helter Skelter"
,"Here, There And Everywhere"
,"Hey Jude"
,"I Saw Her Standing There"
,"I Wanna Be Your Man"
,"I'm Down"
,"I've Got A Feeling"
,"I've Just Seen A Face"
,"Kansas City"
,"Lady Madonna"
,"Let It Be"
,"Long And Winding Road"
,"Long Tall Sally"
,"Magical Mystery Tour"
,"Matchbox"
,"Michelle"
,"Mother Nature's Son"
,"Paperback Writer"
,"Penny Lane"
,"Run For Your Life"
,"Sgt. Pepper's Lonely Hearts Club Band"
,"Sgt. Pepper's Lonely Hearts Club Band"
,"Sgt. Pepper's Lonely Hearts Club Band/The End"
,"She's A Woman"
,"She's Leaving Home"
,"Something"
,"The Fool On The Hill"
,"The Long And Winding Road"
,"Things We Said Today"
,"Think For Yourself "
,"We Can Work It Out"
,"While My Guitar Gently Weeps"
,"Yesterday"
]

pm_songs = pm_songs.loc[~pm_songs['Song'].isin(pm_dup_songs)]
pm_songs.reset_index(inplace = True, drop = True)


#%% Clean-up McCartney songs (2)

pm_internal_dups = [129,320,121,75,338,342,344]
pm_songs = pm_songs.loc[~pm_songs.index.isin(pm_internal_dups)]
pm_songs.reset_index(inplace = True, drop = True)



#%% VADER sentiment

# Hutto, C.J. & Gilbert, E.E. (2014). 
# VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. 
# Eighth International Conference on Weblogs and Social Media (ICWSM-14). 
# Ann Arbor, MI, June 2014.

# see https://github.com/cjhutto/vaderSentiment for usage

# note: depending on how you installed (e.g., using source code download versus pip install),
# you may need to import like this:
# from vaderSentiment import SentimentIntensityAnalyzer
#pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(Song, sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
    print(type(score))
    
    song = []
    compound = []
    neg = []
    neu = []
    pos = []
    
    song.append(Song)
    compound.append(score['compound'])
    neg.append(score['neg'])
    neu.append(score['neu'])
    pos.append(score['pos']) 
    
    score_dict = {'Song' : song
            ,'compound' : compound
            ,'neg' : neg
            ,'neu' : neu
            ,'pos' : pos}
    
        
    scores = pd.DataFrame(score_dict)    
    return scores
    
    
#%% Apply VADER sentiment  to Beatles
    
scoresDF = pd.DataFrame()  #columns = ['Song','compound','neg','neu','pos'])

for row in beatles_songs.itertuples():
    print(row.Song, row.Lyrics)
    scores = sentiment_analyzer_scores(row.Song, row.Lyrics)
    scoresDF = pd.concat([scoresDF,scores])

beatles_songs = beatles_songs.merge(scoresDF, how='inner', on='Song')

print(' mean sentiment for beatles: {:.2f}'.format(beatles_songs.compound.mean()))


#%% Apply VADER sentiment  to Lennon
    
scoresDF = pd.DataFrame() 

for row in jl_songs.itertuples():
    print(row.Song, row.Lyrics)
    scores = sentiment_analyzer_scores(row.Song, row.Lyrics)
    scoresDF = pd.concat([scoresDF,scores])

jl_songs = jl_songs.merge(scoresDF, how='inner', on='Song')

print(' mean sentiment for lenon: {:.2f}'.format(jl_songs.compound.mean()))

#%% Apply VADER sentiment  to McCartney
    
scoresDF = pd.DataFrame() 

for row in pm_songs.itertuples():
    print(row.Song, row.Lyrics)
    scores = sentiment_analyzer_scores(row.Song, row.Lyrics)
    scoresDF = pd.concat([scoresDF,scores])

pm_songs = pm_songs.merge(scoresDF, how='inner', on='Song')

print(' mean sentiment for McCartney: {:.2f}'.format(pm_songs.compound.mean()))


#%% sentiment for all solo

solo_songs = pd.concat([pm_songs, jl_songs])
print(' mean sentiment for all solo: {:.2f}'.format(solo_songs.compound.mean()))

#%% Output everthing to files

jl_songs.to_csv('jl_songs.txt', sep = '\t', encoding = 'utf-8', index = False)
pm_songs.to_csv('pm_songs.txt', sep = '\t', encoding = 'utf-8', index = False)
beatles_songs.to_csv('beatles_songs.txt', sep = '\t', encoding = 'utf-8', index = False)



#%% Take the top20 negative and top20 positive songs from Lennon McCartney solo
# And build a training set
    
jl_songs_vsort = jl_songs.sort_values(by='compound', ascending = True)
jl_songs_vsort.reset_index(inplace=True, drop=True)
#jl_songs.sort_index(inplace = True)
jl_len = len(jl_songs_vsort)-19
jl_neg_songs = jl_songs_vsort.loc[0:19 , ]
jl_pos_songs = jl_songs_vsort.loc[ (jl_len - 19) : jl_len , ]
jl_neg_songs['Label'] = 'n'
jl_pos_songs['Label'] = 'p'

pm_songs_vsort = pm_songs.sort_values(by='compound', ascending = True)
pm_songs_vsort.reset_index(inplace=True, drop=True)
#pm_songs.sort_index(inplace = True)
pm_len = len(pm_songs_vsort)-19
pm_neg_songs = pm_songs_vsort.loc[0:19 , ]
pm_pos_songs = pm_songs_vsort.loc[ (pm_len - 19) : pm_len , ]
pm_neg_songs['Label'] = 'n'
pm_pos_songs['Label'] = 'p'

LM_solo_pos_neg = pd.concat([jl_neg_songs, pm_neg_songs, jl_pos_songs, pm_pos_songs])
LM_solo_pos_neg.reset_index(inplace=True, drop=True)

LM_solo_pos_neg.to_csv('LM_solo_pos_neg.csv', sep='\t', encoding = 'utf-8', index = False)


#%% Get only the beatles songs by Lennon or McCartney and top 20 pos and neg
# for testing

# songs for which the main composer is Lennon OR McCartney
beatles_songs1 = beatles_songs.loc[ (beatles_songs.NumAuth==1) & ( (beatles_songs.L == 1) | (beatles_songs.M == 1) ) ]
beatles_songs1.reset_index(inplace=True, drop=True)

beatles_songs_vsort = beatles_songs1.sort_values(by='compound', ascending = True)
beatles_songs_vsort.reset_index(inplace=True, drop=True)

beatles_len = len(beatles_songs_vsort)-19
beatles_neg_songs = beatles_songs_vsort.loc[0:19 , ]
beatles_pos_songs = beatles_songs_vsort.loc[ (beatles_len - 19) : beatles_len , ]

beatles_neg_songs['Label'] = 'n'
beatles_pos_songs['Label'] = 'p'

LM_Beatles_pos_neg = pd.concat([beatles_neg_songs, beatles_pos_songs])
LM_Beatles_pos_neg.reset_index(inplace=True, drop=True)

LM_Beatles_pos_neg.to_csv('LM_Beatles_pos_neg.csv', sep='\t', encoding = 'utf-8', index = False)


####################################################
## Part 3 Sentiment analysis
####################################################

# Predict the sentiment of the song lyrics for songs from the Beatles, Lennon,
# and McCartney and compare them to the VADER generated sentiment scores


# Files required to run part 3:
# LM_Beatles_pos_neg.csv (generated in part 2)
# LM_solo_pos_neg.csv (generated in part 2)



import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics



#%% Load the file created from song attributions  from 'The Beatles' and lyrics extracted from the api


LM_Beatles_pos_neg = pd.read_csv('LM_Beatles_pos_neg.csv',sep='\t',encoding='utf-8')

LM_solo_pos_neg = pd.read_csv('LM_solo_pos_neg.csv',sep='\t',encoding='utf-8')
    


#%% Combine the dataframes 
# this will be a 66-33 train-test split

Combined_DF = pd.concat([LM_solo_pos_neg,LM_Beatles_pos_neg]) #0-79, 80-119
Combined_DF.reset_index(inplace=True, drop=True)

Combined_DF.loc[Combined_DF['Main composer'] == 'Lennon' , 'Composer'] = 'L'
Combined_DF.loc[Combined_DF['Main composer'] == 'McCartney' , 'Composer'] = 'M'

Combined_DF.reset_index(inplace=True, drop=True)

Combined_DF = Combined_DF.loc[ : , ['Label','Lyrics']]
Combined_DF = Combined_DF.rename( columns = {'Label' : 'Label', 'Lyrics' : 'Review'})

#%%  Remove the labels

DF_labels = Combined_DF.loc[ : , 'Label']
DF_reviews = Combined_DF.loc[ : , 'Review']


#%% Vectorize the data

## Parameters are adjusted here between runs to test a range of possibilities.


########  binary vectorizer (for Bernoulli)

vect_bin = CountVectorizer(input="content",
                        stop_words='english',
                        lowercase = False,
                        binary = True,
                        max_features = 500,
                        #max_df = 0.5,
                        #ngram_range = (1,3)
                        )

########  frequency vectorizer (for NB and SVM)

vect_freq = CountVectorizer(input="content",
                        stop_words='english',
                        lowercase = False,
                        binary = False,
                        max_features = 500,
                        #max_df = 0.5,
                        #ngram_range = (1,3)
                        )

########  TFIDF vectorizer (for NB and SVM)

vect_tfidf = TfidfVectorizer(input="content",
                        stop_words='english',
                        lowercase = False,
                        binary = False,
                        max_features = 500,
                        #max_df = 0.5,
                        #ngram_range = (1,3)
                        )


#%%  Fit the Binary Vect object to the data

transf_bin = vect_bin.fit_transform(DF_reviews)
transf_bin_colnames = vect_bin.get_feature_names()      
transf_bin_DF = pd.DataFrame( transf_bin.toarray(), columns = transf_bin_colnames )
# Add back labels!
transf_bin_DF.insert(loc = 0 , column ='Label', value = DF_labels)



#%%  Fit the Frequency Vect object to the data

transf_freq = vect_freq.fit_transform(DF_reviews)
transf_freq_colnames = vect_freq.get_feature_names() 
transf_freq_DF = pd.DataFrame( transf_freq.toarray(), columns = transf_freq_colnames )
# Add back labels!
transf_freq_DF.insert(loc = 0 , column ='Label', value = DF_labels)


#%% Fit the TFIDF Vect object to the data

transf_tfidf = vect_tfidf.fit_transform(DF_reviews)
transf_tfidf_colnames = vect_tfidf.get_feature_names() 
transf_tfidf_DF = pd.DataFrame( transf_tfidf.toarray(), columns = transf_tfidf_colnames )
# Add back labels!
transf_tfidf_DF.insert(loc = 0 , column ='Label', value = DF_labels)



#%%  Divide the data into train and test sets

#array1 = np.array(range(len(df)))
#Train, Test = train_test_split(array1, test_size=0.2)

BinTrainDF = transf_bin_DF.loc[0:79, :] #0-79, 80-119
FrqTrainDF = transf_freq_DF.loc[0:79, :]
TFIDFTrainDF = transf_tfidf_DF.loc[0:79, :]

BinTestDF = transf_bin_DF.loc[80:119, :]
FrqTestDF = transf_freq_DF.loc[80:119, :]
TFIDFTestDF = transf_tfidf_DF.loc[80:119, :]

#%%  Remove the labels!

## These are for the train and test sets

# Binary
BinTrainDF_labels = BinTrainDF.loc[ : , 'Label']
BinTrainDF = BinTrainDF.drop(axis = 'columns', columns = 'Label')

BinTestDF_labels = BinTestDF.loc[ : , 'Label']
BinTestDF = BinTestDF.drop(axis = 'columns', columns = 'Label')

# Freq
FrqTrainDF_labels = FrqTrainDF.loc[ : , 'Label']
FrqTrainDF = FrqTrainDF.drop(axis = 'columns', columns = 'Label')

FrqTestDF_labels = FrqTestDF.loc[ : , 'Label']
FrqTestDF = FrqTestDF.drop(axis = 'columns', columns = 'Label')

# TFIDF
TFIDFTrainDF_labels = TFIDFTrainDF.loc[ : , 'Label']
TFIDFTrainDF = TFIDFTrainDF.drop(axis = 'columns', columns = 'Label')

TFIDFTestDF_labels = TFIDFTestDF.loc[ : , 'Label']
TFIDFTestDF = TFIDFTestDF.drop(axis = 'columns', columns = 'Label')


## These are for the undivided (total) data

# Binary
BinDF_labels = transf_bin_DF.loc[ : , 'Label']
BinDF = transf_bin_DF.drop(axis = 'columns', columns = 'Label')

# Freq
FrqDF_labels = transf_freq_DF.loc[ : , 'Label']
FrqDF = transf_freq_DF.drop(axis = 'columns', columns = 'Label')  
  
# TFIDF
TFIDF_DF_labels = transf_tfidf_DF.loc[ : , 'Label']
TFIDF_DF = transf_tfidf_DF.drop(axis = 'columns', columns = 'Label')   


#%% Fit Bernoulli NB object to the Binary Vect data

MODEL_NAME = '\n*** NB 1 ***\n '
MODEL= BernoulliNB()

CROSS = BinDF
CROSS_L = BinDF_labels
TRAIN = BinTrainDF
TRAIN_L = BinTrainDF_labels
TEST = BinTestDF
TEST_L = BinTestDF_labels

##...........................................##
print(MODEL_NAME)

# Cross validate within the training data
print(cross_validate(MODEL, CROSS, y = CROSS_L, cv=5))

# Test the model on the 1/3 of data that is Beatles songs
MODEL.fit(TRAIN, TRAIN_L)
Prediction = MODEL.predict(TEST)

print("\nThe prediction is:")
print(Prediction)

print("\nThe actual labels are:")
print(TEST_L)

print('\n Prediction Probabilities \n')
print(np.round(MODEL.predict_proba(TEST),2))       

print('\n Confusion Matrix\n')
print( metrics.confusion_matrix(TEST_L, Prediction) )

print('\n Classification Report \n')
print( metrics.classification_report(TEST_L, Prediction) )


#%%  Fit MNB object to the Frequency Vect data

MODEL_NAME = '\n*** NB 2 ***\n '
MODEL = MultinomialNB()

CROSS = FrqDF
CROSS_L = FrqDF_labels
TRAIN = FrqTrainDF
TRAIN_L = FrqTrainDF_labels
TEST = FrqTestDF
TEST_L = FrqTestDF_labels

##...........................................##
print(MODEL_NAME)

# Cross validate within the training data
print(cross_validate(MODEL, CROSS, y = CROSS_L, cv=5))

# Test the model on the 1/3 of data that is Beatles songs
MODEL.fit(TRAIN, TRAIN_L)
Prediction = MODEL.predict(TEST)

print("\nThe prediction is:")
print(Prediction)

print("\nThe actual labels are:")
print(TEST_L)

print('\n Confusion Matrix\n')
print( metrics.confusion_matrix(TEST_L, Prediction) )

print('\n Classification Report \n')
print( metrics.classification_report(TEST_L, Prediction) )
    

#%%  Fit MNB object to the TFIDF Vect data

MODEL_NAME = '\n*** NB 3 ***\n '
MODEL = MultinomialNB()

CROSS = TFIDF_DF
CROSS_L = TFIDF_DF_labels
TRAIN = TFIDFTrainDF
TRAIN_L = TFIDFTrainDF_labels
TEST = TFIDFTestDF
TEST_L = TFIDFTestDF_labels


##...........................................##
print(MODEL_NAME)

# Cross validate within the training data
print(cross_validate(MODEL, CROSS, y = CROSS_L, cv=5))

# Test the model on the 1/3 of data that is Beatles songs
MODEL.fit(TRAIN, TRAIN_L)
Prediction = MODEL.predict(TEST)

print("\nThe prediction is:")
print(Prediction)

print("\nThe actual labels are:")
print(TEST_L)    

print('\n Confusion Matrix\n')
print( metrics.confusion_matrix(TEST_L, Prediction) )

print('\n Classification Report \n')
print( metrics.classification_report(TEST_L, Prediction) )
    


#%% Function to get the observed log probability of each feature

def log_probability(Colnames, Model):
    
    ColNames = transf_bin_colnames
    
    df_lweights = pd.DataFrame(ColNames, columns=['feature'])
    
    df_lweights.insert(loc=1, column='log_prob1', value=Model.feature_log_prob_[0])
    
    df_lweights.insert(loc=1, column='log_prob2', value=Model.feature_log_prob_[1])
    
    df_lweights['ratio'] = df_lweights.log_prob1 / df_lweights.log_prob2
    
    df_lweights.sort_values(by='ratio', inplace=True, ascending = False)
    
    return df_lweights

#%% get the observed log probabilities 
# feature wights are based on observed freqs and are threfore exactly same for all three models


Colnames = transf_bin_colnames # match column names to model fitted!
MODEL = MODEL # be sure to know which model this is!

weights = log_probability(Colnames, MODEL)


#%% VIS 0

###################################################
##
##   Visualizing the top features
##   Then Visualizing the margin with the top 2 in 2D
##
##########################################################

import matplotlib.pyplot as plt

## Credit: https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d

## Define a function to visualize the TOP words (variables)
def plot_ratios(weights, top_features=10):
    
    
    top_positive_coefficients = weights.iloc[-top_features:,]
    top_negative_coefficients = weights.iloc[:top_features,]
    top_coefficients = pd.concat([top_negative_coefficients, top_positive_coefficients])
    

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ["purple" if c < 1 else "green" for c in top_coefficients.ratio]
    y_pos = np.arange(len(top_coefficients))
    
    ax.barh(y_pos, top_coefficients.ratio, align='center', color=colors)
    feature_names = np.array(top_coefficients.feature)
    ax.set_yticks(np.arange(2 * top_features))
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Ratio')
    ax.set_title('Top Negative and Positive Sentiment Words')
    
    plt.show()

#%% VIS 0 run

features = 10

plot_ratios(weights, features)

#%% SVMS

### NOTE - We CANNOT use SVM directly on the data. 
### SVMs do not run on qualitative data.

#%% SVM 1

MODEL_NAME = '\n*** SVM 1 ***\n '
MODEL = LinearSVC(C=10)

CROSS = BinDF
CROSS_L = BinDF_labels
TRAIN = BinTrainDF
TRAIN_L = BinTrainDF_labels
TEST = BinTestDF
TEST_L = BinTestDF_labels

#CROSS = FrqDF
#CROSS_L = FrqDF_labels
#TRAIN = FrqTrainDF
#TRAIN_L = FrqTrainDF_labels
#TEST = FrqTestDF
#TEST_L = FrqTestDF_labels

#CROSS = TFIDF_DF
#CROSS_L = TFIDF_DF_labels
#TRAIN = TFIDFTrainDF
#TRAIN_L = TFIDFTrainDF_labels
#TEST = TFIDFTestDF
#TEST_L = TFIDFTestDF_labels

##...........................................##
print(MODEL_NAME)

# Cross validate within the training data
print(cross_validate(MODEL, CROSS, y = CROSS_L, cv=5))

# Test the model on the 1/3 of data that is Beatles songs
MODEL.fit(TRAIN, TRAIN_L)
Prediction = MODEL.predict(TEST)

print("\nThe prediction is:")
print(Prediction)

print("\nThe actual labels are:")
print(TEST_L)

print('\n Confusion Matrix\n')
print( metrics.confusion_matrix(TEST_L, Prediction) )

print('\n Classification Report \n')
print( metrics.classification_report(TEST_L, Prediction) )



#%% SVM 2

MODEL_NAME = '\n*** SVM 2 ***\n '
MODEL = SVC(C=10, kernel='rbf', gamma="scale")

CROSS = BinDF
CROSS_L = BinDF_labels
TRAIN = BinTrainDF
TRAIN_L = BinTrainDF_labels
TEST = BinTestDF
TEST_L = BinTestDF_labels

#CROSS = FrqDF
#CROSS_L = FrqDF_labels
#TRAIN = FrqTrainDF
#TRAIN_L = FrqTrainDF_labels
#TEST = FrqTestDF
#TEST_L = FrqTestDF_labels

#CROSS = TFIDF_DF
#CROSS_L = TFIDF_DF_labels
#TRAIN = TFIDFTrainDF
#TRAIN_L = TFIDFTrainDF_labels
#TEST = TFIDFTestDF
#TEST_L = TFIDFTestDF_labels

##...........................................##
print(MODEL_NAME)

# Cross validate within the training data
print(cross_validate(MODEL, CROSS, y = CROSS_L, cv=5))

# Test the model on the 1/3 of data that is Beatles songs
MODEL.fit(TRAIN, TRAIN_L)
Prediction = MODEL.predict(TEST)

print("\nThe prediction is:")
print(Prediction)

print("\nThe actual labels are:")
print(TEST_L) 

print('\n Confusion Matrix\n')
print( metrics.confusion_matrix(TEST_L, Prediction) )

print('\n Classification Report \n')
print( metrics.classification_report(TEST_L, Prediction) )



#%% SVM 3

MODEL_NAME = '\n*** SVM 3 ***\n '
MODEL = SVC(C=10, kernel='poly', degree=2, gamma="scale")

CROSS = BinDF
CROSS_L = BinDF_labels
TRAIN = BinTrainDF
TRAIN_L = BinTrainDF_labels
TEST = BinTestDF
TEST_L = BinTestDF_labels

#CROSS = FrqDF
#CROSS_L = FrqDF_labels
#TRAIN = FrqTrainDF
#TRAIN_L = FrqTrainDF_labels
#TEST = FrqTestDF
#TEST_L = FrqTestDF_labels

#CROSS = TFIDF_DF
#CROSS_L = TFIDF_DF_labels
#TRAIN = TFIDFTrainDF
#TRAIN_L = TFIDFTrainDF_labels
#TEST = TFIDFTestDF
#TEST_L = TFIDFTestDF_labels

##...........................................##
print(MODEL_NAME)

# Cross validate within the training data
print(cross_validate(MODEL, CROSS, y = CROSS_L, cv=5))

# Test the model on the 1/3 of data that is Beatles songs
MODEL.fit(TRAIN, TRAIN_L)
Prediction = MODEL.predict(TEST)

print("\nThe prediction is:")
print(Prediction)

print("\nThe actual labels are:")
print(TEST_L)


print('\n Confusion Matrix\n')
print( metrics.confusion_matrix(TEST_L, Prediction) )

print('\n Classification Report \n')
print( metrics.classification_report(TEST_L, Prediction) )


#%% Get the term weights (SVM coefficients) (NOTE - only works if linear kernal SVM was run last!!)
def get_coefficients (MODEL, colnames = transf_bin_colnames):
    
    coef = MODEL.coef_.ravel()
    coef_df = pd.DataFrame(coef, columns=['coefficient'])
    coef_df.insert(loc = 0, column = 'feature', value = colnames)
    return coef_df


coef_df = get_coefficients(MODEL, transf_bin_colnames)


#%% VIS1 (NOTE - only works for linear kernal!!)

###################################################
##
##   Visualizing the top features
##   Then Visualizing the margin with the top 2 in 2D
##
##########################################################

import matplotlib.pyplot as plt

## Credit: https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d

## Define a function to visualize the TOP words (variables)

def plot_coefficients(MODEL, COLNAMES, top_features):
    
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["purple" if c < 0 else "green" for c in coef[top_coefficients]]
    y_pos = np.arange(len(top_coefficients))
    
    ax.barh(y_pos, coef[top_coefficients], align='center', color=colors)
    feature_names = np.array(COLNAMES)
    ax.set_yticks(np.arange(2 * top_features))
    ax.set_yticklabels(feature_names[top_coefficients])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Coeffient')
    ax.set_title('Top Negative and Positive Sentiment Words')
    
    plt.show()

#%% VIS 1 run (NOTE - only works for linear kernal!!)

model = MODEL
colnames = transf_bin_colnames
features = 10

plot_coefficients(model, colnames, features)


####################################################
## Part 4 Author Predictions
####################################################

# Predict the author of the song lyrics for songs from the Beatles, Lennon,
# and McCartney and compare them to known (or likely) author given by 
# the literature sources.


# Files required to run part 4:
# beatles_songs.txt (generated in part 2)
# jl_songs.txt (generated in part 2)
# pm_songs.txt (generated in part 2)



import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.cluster import KMeans



#%% Load the file created from song attributions  from 'The Beatles' and lyrics extracted from the api


Beatles = pd.read_csv('beatles_songs.txt',sep='\t',encoding='utf-8')
L_solo = pd.read_csv('jl_songs.txt',sep='\t',encoding='utf-8')
M_solo = pd.read_csv('pm_songs.txt',sep='\t',encoding='utf-8') 


#%% prepare 2 LM_beatles datasets (using only l or m) 
# songs for which the main composer is Lennon OR McCartney

L_Beatles = Beatles.loc[ (Beatles.NumAuth==1) & (Beatles.L == 1) ]
L_Beatles.reset_index(inplace=True, drop=True)
L_Beatles = L_Beatles.rename( columns = {'Main composer' : 'Composer'})
L_Beatles.loc[L_Beatles.Composer == 'Lennon', 'Composer'] = 'L'
L_Beatles.loc[L_Beatles.Composer == 'McCartney', 'Composer'] = 'M'

M_Beatles = Beatles.loc[ (Beatles.NumAuth==1) & (Beatles.M == 1) ]
M_Beatles.reset_index(inplace=True, drop=True)
M_Beatles = M_Beatles.rename( columns = {'Main composer' : 'Composer'})
M_Beatles.loc[M_Beatles.Composer == 'Lennon', 'Composer'] = 'L'
M_Beatles.loc[M_Beatles.Composer == 'McCartney', 'Composer'] = 'M'


#%% remove songs from M_solo to equal the number of songs in L_solo

## Subtract 210 songs from M dataset to match number in L dataset


#M_solo_rand = random.sample(range(353), 143)
#may also enter saved list from file here for repeating past results
M_solo_rand = [332, 119, 31, 340, 243, 217, 164, 224, 280, 325, 195, 76, 10, 272, 176, 244, 89, 81, 59, 143, 47, 68, 33, 263, 291, 346, 113, 277, 196, 39, 184, 78, 188, 87, 153, 42, 297, 58, 94, 266, 126, 161, 293, 117, 208, 150, 147, 131, 302, 296, 46, 257, 114, 212, 156, 273, 16, 336, 64, 21, 38, 311, 274, 190, 102, 166, 209, 11, 258, 26, 235, 93, 218, 262, 111, 337, 110, 349, 123, 330, 191, 345, 183, 303, 25, 167, 40, 70, 7, 203, 57, 201, 65, 56, 230, 288, 154, 128, 321, 125, 163, 80, 121, 213, 112, 270, 162, 179, 312, 172, 229, 169, 347, 67, 159, 5, 55, 200, 109, 103, 260, 221, 242, 63, 320, 352, 193, 35, 75, 343, 66, 205, 329, 155, 152, 88, 276, 287, 268, 216, 197, 1, 300]
M_solo = M_solo.loc[M_solo.index.isin(M_solo_rand)]
M_solo.reset_index(inplace = True, drop = True)



#%% Select random samples from each set to compose the test sets
# The test sets will consist of 
# 1) 30 Beatles (15Lennon and 15McCartney) 
# 2) 30 Solo (15Lennon and 15McCartney)


## Solo (Lennon and McCartney) Train and Test
#M_rand = random.sample(range(143), 20)
#L_rand = random.sample(range(143), 20)
#may also enter saved list from file here for repeating past results
M_rand = [47, 1, 5, 87, 97, 15, 39, 92, 35, 12, 88, 10, 126, 11, 90, 46, 110, 120, 2, 53]
#may also enter saved list from file here for repeating past results
L_rand = [110, 54, 118, 1, 48, 102, 15, 99, 136, 46, 61, 23, 85, 55, 142, 40, 33, 56, 60, 13]


M_solo_test = M_solo.iloc[M_rand]
M_solo_test.reset_index(inplace = True, drop = True)

M_solo_train = M_solo[~M_solo.index.isin(M_rand)]
M_solo_train.reset_index(inplace = True, drop = True)

L_solo_test = L_solo.iloc[L_rand]
L_solo_test.reset_index(inplace = True, drop = True)

L_solo_train = L_solo[~L_solo.index.isin(L_rand)]
L_solo_train.reset_index(inplace = True, drop = True)


## Beatles (Lennon and McCartney) Train and Test
#M_beatles_rand = random.sample(range(70), 20)
# may also enter saved list from file here for repeating past results
M_beatles_rand = [13, 46, 20, 68, 55, 34, 29, 56, 54, 6, 47, 32, 52, 59, 30, 3, 2, 50, 33, 28]

## there are two extra Lennon vs. McCartney, take two Lennon out
#L_beatles_rand = random.sample(range(72), 22) 
#L_beatles_rand_small = L_beatles_rand[:]
#print(L_beatles_rand_small)
#subsample = random.sample(range(22), 1)
#print(subsample)
#L_beatles_rand_small.pop(subsample[0])
#subsample = random.sample(range(21), 1)
#print(subsample)
#L_beatles_rand_small.pop(subsample[0])
#print(L_beatles_rand)

#may also enter saved list from file here for repeating past results
L_beatles_rand = [70, 29, 53, 52, 0, 41, 23, 39, 43, 48, 55, 35, 68, 10, 64, 66, 38, 67, 69, 26, 71, 45]
L_beatles_rand_small = [70, 29, 53, 52, 0, 41, 23, 39, 48, 55, 35, 68, 10, 64, 66, 38, 67, 69, 26, 45]

M_Beatles_test = M_Beatles.iloc[M_beatles_rand]
M_Beatles_test.reset_index(inplace = True, drop = True)

M_Beatles_train = M_Beatles[~M_Beatles.index.isin(M_beatles_rand)]
M_Beatles_train.reset_index(inplace = True, drop = True)

L_Beatles_test = L_Beatles.iloc[L_beatles_rand_small]
L_Beatles_test.reset_index(inplace = True, drop = True)

L_Beatles_train = L_Beatles[~L_Beatles.index.isin(L_beatles_rand)]
L_Beatles_train.reset_index(inplace = True, drop = True)


#%% Save random samples for later to a file

#filename = 'rand_samples_2020_06_05b.txt'
#
#randfile = open(filename, 'w')
#
#randfile.write('M_solo_rand\n')
#randfile.write(str(M_solo_rand) + '\n')
#randfile.write('M_rand\n')
#randfile.write(str(M_rand) + '\n')
#randfile.write('L_rand\n')
#randfile.write(str(L_rand) + '\n')
#randfile.write('M_beatles_rand\n')
#randfile.write(str(M_beatles_rand) + '\n')
#randfile.write('L_beatles_rand\n')
#randfile.write(str(L_beatles_rand) + '\n')
#randfile.write('L_beatles_rand_small\n')
#randfile.write(str(L_beatles_rand_small))
#
#randfile.close()

#%% check distribution of random series

#plt.hist(M_solo_rand, normed = False, stacked = False, rwidth = .9)
#plt.title("M_solo_rand")
#plt.show()
#
#plt.hist(M_rand, normed = False, stacked = False, rwidth = .9)
#plt.title("M_rand")
#plt.show()
#
#plt.hist(L_rand, normed = False, stacked = False, rwidth = .9)
#plt.title("L_rand")
#plt.show()
#
#plt.hist(M_beatles_rand, normed = False, stacked = False, rwidth = .9)
#plt.title("M_beatles_rand")
#plt.show()
#
#plt.hist(L_beatles_rand, normed = False, stacked = False, rwidth = .9)
#plt.title("L_beatles_rand")
#plt.show()
#
#plt.hist(L_beatles_rand_small, normed = False, stacked = False, rwidth = .9)
#plt.title("L_beatles_rand_small")
#plt.show()



#%% Combine the dataframes for the different train/test sets

Beatles_train_Beatles_test_DF = pd.concat([ L_Beatles_train, M_Beatles_train, L_Beatles_test, M_Beatles_test ]) #0:49, 50:99, 100:139
Beatles_train_Beatles_test_DF.reset_index(inplace=True, drop=True)
Beatles_train_Beatles_test_meta = Beatles_train_Beatles_test_DF
Beatles_train_Beatles_test_DF = Beatles_train_Beatles_test_DF.loc[ : , ['Composer','Lyrics']]
Beatles_train_Beatles_test_DF = Beatles_train_Beatles_test_DF.rename( columns = {'Composer' : 'Label', 'Lyrics' : 'Review'})
Beatles_train_Beatles_test_start = 100
Beatles_train_Beatles_test_end = 139

solo_train_Beatles_test_DF = pd.concat([ L_solo_train, M_solo_train, L_Beatles_test, M_Beatles_test ]) #0:122, 123:245, 246:285
solo_train_Beatles_test_DF.reset_index(inplace=True, drop=True)
solo_train_Beatles_test_meta = solo_train_Beatles_test_DF
solo_train_Beatles_test_DF = solo_train_Beatles_test_DF.loc[ : , ['Composer','Lyrics']]
solo_train_Beatles_test_DF = solo_train_Beatles_test_DF.rename( columns = {'Composer' : 'Label', 'Lyrics' : 'Review'})
solo_train_Beatles_test_start = 246
solo_train_Beatles_test_end = 285

sb_train_Beatles_test_DF = pd.concat([ L_solo_train, M_solo_train, L_Beatles_train, M_Beatles_train, L_Beatles_test, M_Beatles_test ]) #0:122, 123:245, 246:295, 296:345, 346:385
sb_train_Beatles_test_DF.reset_index(inplace=True, drop=True)
sb_train_Beatles_test_meta = sb_train_Beatles_test_DF
sb_train_Beatles_test_DF = sb_train_Beatles_test_DF.loc[ : , ['Composer','Lyrics']]
sb_train_Beatles_test_DF = sb_train_Beatles_test_DF.rename( columns = {'Composer' : 'Label', 'Lyrics' : 'Review'})
sb_train_Beatles_test_start = 346
sb_train_Beatles_test_end = 385

solo_train_solo_test_DF = pd.concat([ L_solo_train, M_solo_train, L_solo_test, M_solo_test ]) #0:122, 123:245, 246:285
solo_train_solo_test_DF.reset_index(inplace=True, drop=True)
solo_train_solo_test_meta = solo_train_solo_test_DF
solo_train_solo_test_DF = solo_train_solo_test_DF.loc[ : , ['Composer','Lyrics']]
solo_train_solo_test_DF = solo_train_solo_test_DF.rename( columns = {'Composer' : 'Label', 'Lyrics' : 'Review'})
solo_train_solo_test_start = 246
solo_train_solo_test_end = 285

#%%   Output the dataframes to csv

#Beatles_train_Beatles_test_DF.to_csv('Beatles_train_Beatles_test_DF.txt',sep='\t', index = False, encoding='utf-8')
#Beatles_train_Beatles_test_meta.to_csv('Beatles_train_Beatles_test_meta.txt',sep='\t', index = False, encoding='utf-8')

#solo_train_Beatles_test_DF.to_csv('solo_train_Beatles_test_DF.txt',sep='\t', index = False, encoding='utf-8')
#solo_train_Beatles_test_meta.to_csv('solo_train_Beatles_test_meta.txt',sep='\t', index = False, encoding='utf-8')

#sb_train_Beatles_test_DF.to_csv('sb_train_Beatles_test_DF.txt',sep='\t', index = False, encoding='utf-8')
#sb_train_Beatles_test_meta.to_csv('sb_train_Beatles_test_meta.txt',sep='\t', index = False, encoding='utf-8')

#solo_train_solo_test_DF.to_csv('solo_train_solo_test_DF.txt',sep='\t', index = False, encoding='utf-8')
#solo_train_solo_test_meta.to_csv('solo_train_solo_test_meta.txt',sep='\t', index = False, encoding='utf-8')



#%%  ###### Run the Combined_DF ######

# This is where the dataset from above is selected.

# It is also where to start testing different DTM-Model combinations
# without re-running any of the code above

# All the code below this will use the dataset selected here.

# The code below should be deterministic for both the cross-validated
# and out-of sample test sets if the random samples above were not changed. 

# <<SELECT>>

#Combined_DF = Beatles_train_Beatles_test_DF
#Combined_DF = solo_train_Beatles_test_DF
#Combined_DF = sb_train_Beatles_test_DF
Combined_DF = solo_train_solo_test_DF

#%%   Remove the labels

DF_labels = Combined_DF.loc[ : , 'Label']
DF_reviews = Combined_DF.loc[ : , 'Review']



#%% Vectors (Binary, Frequency, TFIDF)

# <<SELECT>>

## Parameters are adjusted here between runs to test a range of possibilities.

## Adjust DTM parameters here:

#STOP_WORDS = 'english'
STOP_WORDS = None

#LOWER_CASE = True
LOWER_CASE = False

#MAX_FEATURES = 500
MAX_FEATURES = 5000

NGRAM_RANGE = (1,1)
#NGRAM_RANGE = (1,3)



########  binary vectorizer (for Bernoulli)

vect_bin = CountVectorizer(input="content",
                        stop_words = STOP_WORDS,
                        lowercase = LOWER_CASE,
                        binary = True,
                        max_features = MAX_FEATURES,
                        ngram_range = NGRAM_RANGE
                        )

########  frequency vectorizer (for NB and SVM)

vect_freq = CountVectorizer(input="content",
                        stop_words = STOP_WORDS,
                        lowercase = LOWER_CASE,
                        binary = False,
                        max_features = MAX_FEATURES,
                        ngram_range = NGRAM_RANGE
                        )

########  TFIDF vectorizer (for NB and SVM)

vect_tfidf = TfidfVectorizer(input="content",
                        stop_words = STOP_WORDS,
                        lowercase = LOWER_CASE,
                        binary = False,
                        max_features = MAX_FEATURES,
                        ngram_range = NGRAM_RANGE
                        )

    


#%%  Fit the Binary Vect object to the data

transf_bin = vect_bin.fit_transform(DF_reviews)
transf_bin_colnames = vect_bin.get_feature_names()      
transf_bin_DF = pd.DataFrame( transf_bin.toarray(), columns = transf_bin_colnames )
# Add back labels!
transf_bin_DF.insert(loc = 0 , column ='Label', value = DF_labels)



#%%  Fit the Frequency Vect object to the data

transf_freq = vect_freq.fit_transform(DF_reviews)
transf_freq_colnames = vect_freq.get_feature_names() 
transf_freq_DF = pd.DataFrame( transf_freq.toarray(), columns = transf_freq_colnames )
# Add back labels!
transf_freq_DF.insert(loc = 0 , column ='Label', value = DF_labels)


#%% Fit the TFIDF Vect object to the data

transf_tfidf = vect_tfidf.fit_transform(DF_reviews)
transf_tfidf_colnames = vect_tfidf.get_feature_names() 
transf_tfidf_DF = pd.DataFrame( transf_tfidf.toarray(), columns = transf_tfidf_colnames )
# Add back labels!
transf_tfidf_DF.insert(loc = 0 , column ='Label', value = DF_labels)



#%%  Divide the data into train and test sets

# These should be selected depending on the train/test set chosen above

# <<SELECT>>

#test_start = Beatles_train_Beatles_test_start #100
#test_end = Beatles_train_Beatles_test_end #139

#test_start = solo_train_Beatles_test_start #246
#test_end = solo_train_Beatles_test_end #285

#test_start = sb_train_Beatles_test_start #346
#test_end = sb_train_Beatles_test_end #385

test_start = solo_train_solo_test_start #246
test_end = solo_train_solo_test_end #285

print('Does this agree?')
print(test_start)
print(test_end)

BinTrainDF = transf_bin_DF.loc[0:test_start-1, :] 
FrqTrainDF = transf_freq_DF.loc[0:test_start-1, :]
TFIDFTrainDF = transf_tfidf_DF.loc[0:test_start-1, :]

BinTestDF = transf_bin_DF.loc[test_start:test_end, :]
FrqTestDF = transf_freq_DF.loc[test_start:test_end, :]
TFIDFTestDF = transf_tfidf_DF.loc[test_start:test_end, :]



#%%  Remove those labels!
## These are for the train and test sets

# Binary
BinTrainDF_labels = BinTrainDF.loc[ : , 'Label']
BinTrainDF = BinTrainDF.drop(axis = 'columns', columns = 'Label')

BinTestDF_labels = BinTestDF.loc[ : , 'Label']
BinTestDF = BinTestDF.drop(axis = 'columns', columns = 'Label')

# Freq
FrqTrainDF_labels = FrqTrainDF.loc[ : , 'Label']
FrqTrainDF = FrqTrainDF.drop(axis = 'columns', columns = 'Label')

FrqTestDF_labels = FrqTestDF.loc[ : , 'Label']
FrqTestDF = FrqTestDF.drop(axis = 'columns', columns = 'Label')

# TFIDF
TFIDFTrainDF_labels = TFIDFTrainDF.loc[ : , 'Label']
TFIDFTrainDF = TFIDFTrainDF.drop(axis = 'columns', columns = 'Label')

TFIDFTestDF_labels = TFIDFTestDF.loc[ : , 'Label']
TFIDFTestDF = TFIDFTestDF.drop(axis = 'columns', columns = 'Label')

#%%  Remove those labels!
## These are for the undivided (total) data used for cross-validations

# Binary
BinDF_labels = transf_bin_DF.loc[ : , 'Label']
BinDF = transf_bin_DF.drop(axis = 'columns', columns = 'Label')

# Freq
FrqDF_labels = transf_freq_DF.loc[ : , 'Label']
FrqDF = transf_freq_DF.drop(axis = 'columns', columns = 'Label')  
  
# TFIDF
TFIDF_DF_labels = transf_tfidf_DF.loc[ : , 'Label']
TFIDF_DF = transf_tfidf_DF.drop(axis = 'columns', columns = 'Label')   


#%% Cluster the matrix
#
#kmeans_object = KMeans(n_clusters=8)
##print(kmeans_object)
#kmeans_object.fit(FrqDF)
## Get cluster assignment labels
#labels = kmeans_object.labels_
#print(labels)
## Format results as a DataFrame
#Myresults = pd.DataFrame([FrqDF_labels,labels]).T
#print(Myresults)
#
#inertia = kmeans_object.inertia_
#print(inertia)

#%% Fit Bernoulli NB object to the Binary Vect data

MODEL_NAME = '\n*** NB 1 ***\n '
MODEL= BernoulliNB()

CROSS = BinDF
CROSS_L = BinDF_labels
TRAIN = BinTrainDF
TRAIN_L = BinTrainDF_labels
TEST = BinTestDF
TEST_L = BinTestDF_labels

##...........................................##
print(MODEL_NAME)

# Cross validate within the training data
c_val = cross_validate(MODEL, CROSS, y = CROSS_L, cv=6)
#c_val = cross_validate(MODEL, CROSS, y = CROSS_L, cv=6, return_estimator=True)
#my_classes = (c_val['estimator'])
print(c_val)
print(c_val['test_score'])
print('Average {:.2f}'.format(np.average(c_val['test_score'])))


# Test the model on the out-of-sample population using the full training set
MODEL.fit(TRAIN, TRAIN_L)
Prediction = MODEL.predict(TEST)

print("\nThe prediction is:")
print(Prediction)

print("\nThe actual labels are:")
print(TEST_L)

print('\n Prediction Probabilities \n')
print(np.round(MODEL.predict_proba(TEST),2))       

print('\n Confusion Matrix\n')
print( metrics.confusion_matrix(TEST_L, Prediction) )

print(MODEL_NAME)
print('\n Classification Report \n')
print( metrics.classification_report(TEST_L, Prediction) )

print('Cross Val Average {:.2f}'.format(np.average(c_val['test_score'])))



#%%  Fit MNB object to the Frequency Vect data

MODEL_NAME = '\n*** NB 2 ***\n '
MODEL = MultinomialNB()

CROSS = FrqDF
CROSS_L = FrqDF_labels
TRAIN = FrqTrainDF
TRAIN_L = FrqTrainDF_labels
TEST = FrqTestDF
TEST_L = FrqTestDF_labels

##...........................................##
print(MODEL_NAME)

# Cross validate within the training data
c_val = cross_validate(MODEL, CROSS, y = CROSS_L, cv=6)
print(c_val)
print(c_val['test_score'])
print('Average {:.2f}'.format(np.average(c_val['test_score'])))

# Test the model on the out-of-sample population using the full training set
MODEL.fit(TRAIN, TRAIN_L)
Prediction = MODEL.predict(TEST)

print("\nThe prediction is:")
print(Prediction)

print("\nThe actual labels are:")
print(TEST_L)

#print('\n Prediction Probabilities \n')
#print(np.round(MODEL.predict_proba(TEST),2))       

print('\n Confusion Matrix\n')
print( metrics.confusion_matrix(TEST_L, Prediction) )

print(MODEL_NAME)
print('\n Classification Report \n')
print( metrics.classification_report(TEST_L, Prediction) )

print('Cross Val Average {:.2f}'.format(np.average(c_val['test_score'])))  

#%%  Fit MNB object to the TFIDF Vect data

MODEL_NAME = '\n*** NB 3 ***\n '
MODEL = MultinomialNB()

CROSS = TFIDF_DF
CROSS_L = TFIDF_DF_labels
TRAIN = TFIDFTrainDF
TRAIN_L = TFIDFTrainDF_labels
TEST = TFIDFTestDF
TEST_L = TFIDFTestDF_labels


##...........................................##
print(MODEL_NAME)

# Cross validate within the training data
c_val = cross_validate(MODEL, CROSS, y = CROSS_L, cv=6)
print(c_val)
print(c_val['test_score'])
print('Average {:.2f}'.format(np.average(c_val['test_score'])))


# Test the model on the out-of-sample population using the full training set
MODEL.fit(TRAIN, TRAIN_L)
Prediction = MODEL.predict(TEST)

print("\nThe prediction is:")
print(Prediction)

print("\nThe actual labels are:")
print(TEST_L)

print('\n Prediction Probabilities \n')
print(np.round(MODEL.predict_proba(TEST),2))       

print('\n Confusion Matrix\n')
print( metrics.confusion_matrix(TEST_L, Prediction) )

print(MODEL_NAME)
print('\n Classification Report \n')
print( metrics.classification_report(TEST_L, Prediction) )
    
print('Cross Val Average {:.2f}'.format(np.average(c_val['test_score'])))






#%% Function to get the observed log probability of each feature for NB

def log_probability(Colnames, Model):
    
    ColNames = transf_bin_colnames
    
    df_lweights = pd.DataFrame(ColNames, columns=['feature'])
    
    df_lweights.insert(loc=1, column='log_prob1', value=Model.feature_log_prob_[0])
    
    df_lweights.insert(loc=1, column='log_prob2', value=Model.feature_log_prob_[1])
    
    df_lweights['ratio'] = df_lweights.log_prob1 / df_lweights.log_prob2
    
    df_lweights.sort_values(by='ratio', inplace=True, ascending = False)
    
    return df_lweights

#%% get the observed log probabilities for NB
    
# feature wights are based on observed freqs and are threfore exactly same for all three models

# <<SELECT>>
    
Colnames = transf_tfidf_colnames # match column names to "Fit the Vect object to the data" above
MODEL = MODEL # be sure to know which NB model this is! (the last one that was run above)

weights = log_probability(Colnames, MODEL)



#%% VIS 0 for NB

###################################################
##
##   Visualizing the top features
##   Then Visualizing the margin with the top 2 in 2D
##
##########################################################

import matplotlib.pyplot as plt

## Credit: https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d

## Define a function to visualize the TOP words (variables)
def plot_ratios(weights, top_features=10):
    
    
    top_positive_coefficients = weights.iloc[-top_features:,]
    top_negative_coefficients = weights.iloc[:top_features,]
    top_coefficients = pd.concat([top_negative_coefficients, top_positive_coefficients])
    
    
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = ["purple" if c < 1 else "green" for c in top_coefficients.ratio]
    y_pos = np.arange(len(top_coefficients))
    
    ax.barh(y_pos, top_coefficients.ratio, align='center', color=colors)
    feature_names = np.array(top_coefficients.feature)
    ax.set_yticks(np.arange(2 * top_features))
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Ratio')
    #ax.set_title('Top Negative and Positive Sentiment Words')
    ax.set_title('Top 10 McCartney and Lennon terms (log ratios)')
    
    plt.show()

#%% VIS 0 for NB run

features = 10

plot_ratios(weights, features)

#%% SVMS

### NOTE - SVMs do not run on qualitative data.

#%% SVM 1

MODEL_NAME = '\n*** SVM 1 ***\n '
MODEL = LinearSVC(C=1)

# <<SELECT>>

#CROSS = BinDF
#CROSS_L = BinDF_labels
#TRAIN = BinTrainDF
#TRAIN_L = BinTrainDF_labels
#TEST = BinTestDF
#TEST_L = BinTestDF_labels

#CROSS = FrqDF
#CROSS_L = FrqDF_labels
#TRAIN = FrqTrainDF
#TRAIN_L = FrqTrainDF_labels
#TEST = FrqTestDF
#TEST_L = FrqTestDF_labels

CROSS = TFIDF_DF
CROSS_L = TFIDF_DF_labels
TRAIN = TFIDFTrainDF
TRAIN_L = TFIDFTrainDF_labels
TEST = TFIDFTestDF
TEST_L = TFIDFTestDF_labels

##...........................................##
print(MODEL_NAME)

# Cross validate within the training data
c_val = cross_validate(MODEL, CROSS, y = CROSS_L, cv=6)
print(c_val)
print(c_val['test_score'])
print('Average {:.2f}'.format(np.average(c_val['test_score'])))

# Test the model on the out-of-sample population using the full training set
MODEL.fit(TRAIN, TRAIN_L)
Prediction = MODEL.predict(TEST)

print("\nThe prediction is:")
print(Prediction)

print("\nThe actual labels are:")
print(TEST_L)

#print('\n Prediction Probabilities \n')
#print(np.round(MODEL.predict_proba(TEST),2))       

print('\n Confusion Matrix\n')
print( metrics.confusion_matrix(TEST_L, Prediction) )

print(MODEL_NAME)
print('\n Classification Report \n')
print( metrics.classification_report(TEST_L, Prediction) )

print('Cross Val Average {:.2f}'.format(np.average(c_val['test_score'])))

#%% SVM 2

MODEL_NAME = '\n*** SVM 2 ***\n '
MODEL = SVC(C=10, kernel='rbf', gamma="scale") 

# <<SELECT>>

CROSS = BinDF
CROSS_L = BinDF_labels
TRAIN = BinTrainDF
TRAIN_L = BinTrainDF_labels
TEST = BinTestDF
TEST_L = BinTestDF_labels

#CROSS = FrqDF
#CROSS_L = FrqDF_labels
#TRAIN = FrqTrainDF
#TRAIN_L = FrqTrainDF_labels
#TEST = FrqTestDF
#TEST_L = FrqTestDF_labels

#CROSS = TFIDF_DF
#CROSS_L = TFIDF_DF_labels
#TRAIN = TFIDFTrainDF
#TRAIN_L = TFIDFTrainDF_labels
#TEST = TFIDFTestDF
#TEST_L = TFIDFTestDF_labels

##...........................................##
print(MODEL_NAME)

# Cross validate within the training data
c_val = cross_validate(MODEL, CROSS, y = CROSS_L, cv=6)
print(c_val)
print(c_val['test_score'])
print('Average {:.2f}'.format(np.average(c_val['test_score'])))

# Test the model on the out-of-sample population using the full training set
MODEL.fit(TRAIN, TRAIN_L)
Prediction = MODEL.predict(TEST)

print("\nThe prediction is:")
print(Prediction)

print("\nThe actual labels are:")
print(TEST_L)

#print('\n Prediction Probabilities \n')
#print(np.round(MODEL.predict_proba(TEST),2))       

print('\n Confusion Matrix\n')
print( metrics.confusion_matrix(TEST_L, Prediction) )

print(MODEL_NAME)
print('\n Classification Report \n')
print( metrics.classification_report(TEST_L, Prediction) )

print('Cross Val Average {:.2f}'.format(np.average(c_val['test_score'])))

#%% SVM 3

MODEL_NAME = '\n*** SVM 3 ***\n '
MODEL = SVC(C=1, kernel='poly', degree=2, gamma="scale")

# <<SELECT>>

CROSS = BinDF
CROSS_L = BinDF_labels
TRAIN = BinTrainDF
TRAIN_L = BinTrainDF_labels
TEST = BinTestDF
TEST_L = BinTestDF_labels

#CROSS = FrqDF
#CROSS_L = FrqDF_labels
#TRAIN = FrqTrainDF
#TRAIN_L = FrqTrainDF_labels
#TEST = FrqTestDF
#TEST_L = FrqTestDF_labels

#CROSS = TFIDF_DF
#CROSS_L = TFIDF_DF_labels
#TRAIN = TFIDFTrainDF
#TRAIN_L = TFIDFTrainDF_labels
#TEST = TFIDFTestDF
#TEST_L = TFIDFTestDF_labels

##...........................................##
print(MODEL_NAME)

# Cross validate within the training data
c_val = cross_validate(MODEL, CROSS, y = CROSS_L, cv=6)
print(c_val)
print(c_val['test_score'])
print('Average {:.2f}'.format(np.average(c_val['test_score'])))

# Test the model on the out-of-sample population using the full training set
MODEL.fit(TRAIN, TRAIN_L)
Prediction = MODEL.predict(TEST)

print("\nThe prediction is:")
print(Prediction)

print("\nThe actual labels are:")
print(TEST_L)

#print('\n Prediction Probabilities \n')
#print(np.round(MODEL.predict_proba(TEST),2))       


print('\n Confusion Matrix\n')
print( metrics.confusion_matrix(TEST_L, Prediction) )

print(MODEL_NAME)
print('\n Classification Report \n')
print( metrics.classification_report(TEST_L, Prediction) )

print('Cross Val Average {:.2f}'.format(np.average(c_val['test_score'])))

#%% Function to get the SVM term weights (SVM coefficients) (NOTE - only works for linear kernal!!)

def get_coefficients (MODEL, colnames = transf_bin_colnames):
    
    coef = MODEL.coef_.ravel()
    coef_df = pd.DataFrame(coef, columns=['coefficient'])
    coef_df.insert(loc = 0, column = 'feature', value = colnames)
    return coef_df


#%% Run Function to get the SVM term weights

# <<SELECT>>

# Note - this only works if last SVM model run was the linear kernal.    

Colnames = transf_tfidf_colnames # match column names to "Fit the Vect object to the data" above
MODEL = MODEL # be sure to know which model this is! (the last one that was run above)


coef_df = get_coefficients(MODEL, Colnames)
#coef_df.to_csv('coefficients.csv', sep=',', index=False)

#%% VIS1 SVMs (NOTE - only works for linear kernal!!)

###################################################
##
##   Visualizing the top features
##   Then Visualizing the margin with the top 2 in 2D
##
##########################################################

import matplotlib.pyplot as plt

## Credit: https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d

## Define a function to visualize the TOP words (variables)

def plot_coefficients(MODEL, COLNAMES, top_features):
    
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    
#    # create plot
#    plt.figure(figsize=(15, 5))
#    colors = ["purple" if c < 0 else "green" for c in coef[top_coefficients]]
#    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
#    feature_names = np.array(COLNAMES)
#    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=90, ha="right")
#    plt.show()
    
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["purple" if c < 0 else "green" for c in coef[top_coefficients]]
    y_pos = np.arange(len(top_coefficients))
    
    ax.barh(y_pos, coef[top_coefficients], align='center', color=colors)
    feature_names = np.array(COLNAMES)
    ax.set_yticks(np.arange(2 * top_features))
    ax.set_yticklabels(feature_names[top_coefficients])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Coeffient')
    #ax.set_title('Top Negative and Positive Sentiment Words')
    ax.set_title('Top 10 McCartney and Lennon terms for SVM')
    
    plt.show()

#%% VIS 1 SVMs run (NOTE - only works for linear kernal!!)

# <<SELECT>>
    
Colnames = transf_tfidf_colnames # match column names to "Fit the Vect object to the data" above
MODEL = MODEL # be sure to know which model this is! (the last one that was run above)
features = 10

plot_coefficients(MODEL, Colnames, features)

