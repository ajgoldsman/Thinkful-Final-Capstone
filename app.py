# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:39:35 2019

@author: Amichai
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials #To access authorised Spotify data

client_id = '88ea7b34d3cb422193552956b5668c55'
client_secret = '60e85b9e559a4eaca1e2de6ca1d9dd42'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) #spotify object to access API

import lyricsgenius as genius
genius_token = 'A4WgAMQRzJCbqvs-yEzRUaeODpiURx6QdYWjSjGiAnxIr4QoCmPAAzpxu1MlDrFY'
api = genius.Genius(genius_token)


import spacy
nlp = spacy.load('en')

#Constants:
ordered_str_keys = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
ordered_str_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

sp_cols = ['album-album_type', 'album-name', 'album-release_date', 'artists-name', 'track-name']
first_feats = ['album_type', 'album_name', 'album_release_date', 'artists_name', 'track_name']
second_feats = ['danceability', 'key', 'loudness', 'mode', 'speechiness', 'liveness', 'tempo', 'duration_ms', 'time_signature']
        

def uniqueSorting(data):
    uniques = pd.DataFrame()
    uni_col = []
    num_uni = []
    avgdiff_uni = []

    for col in list(data.columns):
        try:
            uni_col.append(list(np.unique(data[col])))
            num_uni.append(len(np.unique(data[col])))
        except:
            uni_col.append(list((data[col]).unique()[pd.notnull(list((data[col]).unique()))]))
            num_uni.append(len((data[col]).unique()[pd.notnull(list((data[col]).unique()))]))

        try:
            avgdiff_uni.append(np.mean(np.diff(np.unique(data[col]))))
        except: 
            avgdiff_uni.append('N/A')

    uniques['Category'] = list(data.columns)
    uniques['Unique Values'] = uni_col
    uniques['Num Uniques'] = num_uni
    uniques['Avg Diff Among Uniques'] = avgdiff_uni
    
    return uniques

def catSorting(data):
    uniques = uniqueSorting(data)
    
    drop_cols = [uniques['Category'][i] for i in range(len(uniques)) if uniques['Num Uniques'][i] == 1]
    data = data.drop(drop_cols, axis=1)
    
    uniques = uniqueSorting(data)
    
    all_cols = [item for item in list(data.columns) if item.split('_')[0]!='lyric']
    
    #Separate features by type
    #String Categoricals:
    str_cat_cols = []
    for col in all_cols: 
        if (list(uniques[uniques['Category']==col]['Avg Diff Among Uniques'])[0] == 'N/A'):
            str_cat_cols.append(col)

    #Binary Categoricals:
    bin_cat_cols = [item for item in list(data.columns) if item.split('_')[0]=='lyric']
    for col in all_cols:
        if (col not in str_cat_cols):   
            if (list(uniques[uniques['Category']==col]['Num Uniques'])[0] == 2):
                bin_cat_cols.append(col)

    #Numerical Categorical features:
    cat_cols = []
    for col in all_cols:
        if ((col not in str_cat_cols) and (col not in bin_cat_cols)):   
            if ((list(uniques[uniques['Category']==col]['Avg Diff Among Uniques'])[0] < 1.5) and (list(uniques[uniques['Category']==col]['Num Uniques'])[0] < 60)):
                cat_cols.append(col)

    #Continuous features:
    cont_cols = []
    for col in all_cols:
        if ((col not in cat_cols) and (col not in str_cat_cols) and (col not in bin_cat_cols)):
            cont_cols.append(col)

    #print('String Categorical Features: \n', str_cat_cols)

    #print('\nBinary Dummy Categorical Features: \n', bin_cat_cols)

    #print('\nNumerical Categorical Features: \n', cat_cols)

    #print('\nContinuous Features: \n', cont_cols)
    
    return data, str_cat_cols, bin_cat_cols, cat_cols, cont_cols

def text_cleaner(text):
    import re
    # Visual inspection identifies a form of punctuation spaCy does not
    # recognize: the double dash '--'.  Better get rid of it now!
    text = str(text)
    text = re.sub('--',' ',text)
    text = re.sub("[\[].*?[\]]", "", text)
    text = ' '.join(text.lower().split())
    return text

from difflib import SequenceMatcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()



rf = pickle.load(open('musicclassifier/pkl_objects/classifier.pkl', 'rb'))
  

app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')
#app.config['EXPLAIN_TEMPLATE_LOADING'] = True

valid_song = 'False'
error_bin = 'False'

@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

#@app.route('/verify', methods=['POST', 'GET'])
#def verify():
#    valid_song = 'False'
#    error_bin = 'False'
#    if request.method == 'POST':
#        track = request.form['track_name']
#        artist = request.form['artists_name']
#        
#        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
#        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) #spotify object to access API
#
#        result = sp.search(q='artist:'+ artist + ' track:' + track)
#        try:
#            [i for i in range(10) if result['tracks']['items'][i]['name'] == track][0]
#        except:
#            error_bin = 'True'
#            
#        if error_bin == 'False':
#            try:
#                api = genius.Genius(genius_token)
#                song = api.search_song(track.split(' - From ')[0].split(' (From')[0], artist)
#                if (((song.artist.replace("\'","’").lower() not in artist.replace("\'","’").lower()) and (artist.replace("\'","’").lower() not in song.artist.replace("\'","’").lower()) and (similar(song.artist.replace("\'","’").lower(), artist.replace("\'","’").lower())<0.75)) or 
#                     ((song.title.replace("\'","’").lower() not in track.replace("\'","’").lower()) and (track.replace("\'","’").lower() not in song.title.replace("\'","’").lower()) and (similar(song.title.replace("\'","’").lower(), track.replace("\'","’").lower())<0.75))):
#                    error_bin = 'True'
#                else:
#                    valid_song = 'True'
#                    error_bin = 'False'
#            except:
#                error_bin = 'True'
#                
#    if valid_song == 'False':
#        return redirect(url_for('home'))
#    else:
#        return redirect(url_for('predict'))
    
   
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        track = request.form['track_name']
        artist = request.form['artists_name']
        df = pd.read_csv('clustered_kids_df.csv')
        df, str_cat_cols, bin_cat_cols, cat_cols, cont_cols = catSorting(df)
        
        album_name_cats = [item.replace('album_name_cat_', '') for item in bin_cat_cols if len(item.split('album_name_cat_'))>1 ]
        track_name_cats = [item.replace('track_name_cat_', '') for item in bin_cat_cols if len(item.split('track_name_cat_'))>1 ]
        lyric_strs = [item.replace('lyric_', '') for item in bin_cat_cols if len(item.split('lyric_'))>1 ]

        result = sp.search(q='artist:'+ artist + ' track:' + track)
        try:
            ind = [i for i in range(10) if result['tracks']['items'][i]['name'] == track][0]
        except:
            ind = 0
            
        try:
            item = result['tracks']['items'][ind]
        except:
            return render_template('error.html')
            #print('Song not found. Please try with a different song.')
            
        new_song_df = pd.DataFrame(0, index=np.arange(0, 1), columns = list(df.columns)) ##Make a dataframe
        new_song_df['track_name'] = [item['name']]
        try:
            new_song_df['artists_name'] = [item['artists']['name']]
        except:
            new_song_df['artists_name'] = [item['artists'][0]['name']]       

        try:
            features = sp.audio_features(item['uri'])[0]     #extract audio_features_df    
            #print('Audio Features extraction successful.')
            
        except:
             return render_template('error.html')
#            print('Audio Features extraction not successful. Please try with a different song.')
#            for key in second_feats:
#                new_song_df[key] = np.nan
        
        try:
            song = api.search_song(new_song_df['track_name'][0].split(' - ')[0], new_song_df['artists_name'][0])
            if (((song.artist.replace("\'","’").lower() in new_song_df['artists_name'][0].replace("\'","’").lower()) or (new_song_df['artists_name'][0].replace("\'","’").lower() in song.artist.replace("\'","’").lower()) or (similar(song.artist.replace("\'","’").lower(), new_song_df['artists_name'][0].replace("\'","’").lower())>0.75)) or 
                 ((song.title.replace("\'","’").lower() in new_song_df['track_name'][0].replace("\'","’").lower()) or (new_song_df['track_name'][0].replace("\'","’").lower() in song.title.replace("\'","’").lower()) or (similar(song.title.replace("\'","’").lower(), new_song_df['track_name'][0].replace("\'","’").lower())>0.75))):
                songlyrics = song.lyrics
            else:
                try:
                    song = api.search_song(track, artist)
                    if (((song.artist.replace("\'","’").lower() in artist.replace("\'","’").lower()) or (artist.replace("\'","’").lower() in song.artist.replace("\'","’").lower()) or (similar(song.artist.replace("\'","’").lower(), artist.replace("\'","’").lower())>0.75)) or 
                         ((song.title.replace("\'","’").lower() in track.replace("\'","’").lower()) or (track.replace("\'","’").lower() in song.title.replace("\'","’").lower()) or (similar(song.title.replace("\'","’").lower(), track.replace("\'","’").lower())>0.75))):
                        songlyrics = song.lyrics
                    else:
                        return render_template('error.html')
                except:
                    return render_template('error.html')
        except:
            return render_template('error.html')
        
        
        for i in range(len(sp_cols)):   #extract song_df
            components = sp_cols[i].split('-')
            if len(components)==2:
                if components[0] == 'track':
                    new_song_df[first_feats[i]] = [item[components[1]]]
                else:
                    try:
                        new_song_df[first_feats[i]] = [item[components[0]][components[1]]]
                    except:
                        new_song_df[first_feats[i]] = [item[components[0]][0][components[1]]]       
        
            elif len(components)==3:    
                try:
                    new_song_df[first_feats[i]] = [item[components[0]][components[1]][components[2]]]
                except:
                    new_song_df[first_feats[i]] = [item[components[0]][components[1]][0][components[2]]]       
        
        new_song_df['album_release_year'] = [int(new_song_df['album_release_date'][0].split('-')[0])]
        try:
            new_song_df['album_release_month'] = [int(new_song_df['album_release_date'][0].split('-')[1])]
        except:
            new_song_df['album_release_month'] = [1]
        
        for key in second_feats:
            new_song_df[key] = features[key]
                    
        new_song_df['mode_Major'] = [new_song_df['mode'][0]]
        new_song_df = new_song_df.drop(['mode', 'album_release_date'], axis=1)
        
                    
        new_song_df['lyrics'] = songlyrics            
            
        if new_song_df['album_type'][0]!='compilation':
            new_song_df['album_type_' + new_song_df['album_type'][0]] = [1]
        
        if new_song_df['album_release_month'][0] != 1:
            new_song_df['album_release_month_' + ordered_str_months[new_song_df['album_release_month'][0]-1]] = [1]
        
        if new_song_df['time_signature'][0] != 1:
            new_song_df['time_signature_' + str(new_song_df['time_signature'][0]) + '.0'] = [1]
        
        if new_song_df['key'][0] != 9:
            new_song_df['key_' + ordered_str_keys[new_song_df['key'][0]]] = [1]
        
        for cat in album_name_cats:
            if cat in new_song_df['album_name'][0]:
                new_song_df['album_name_cat_' + cat] = [1]
                
        for cat in track_name_cats:
            if cat in new_song_df['track_name'][0]:
                new_song_df['track_name_cat_' + cat] = [1]
                
        
        words = [token.lemma_ for token in nlp(text_cleaner(new_song_df['lyrics'][0])) if token.lemma_ in lyric_strs]
    
        for word in words:
            new_song_df.loc[0, 'lyric_'+word] += 1
        
        #NLP tokens
        lyr = nlp(text_cleaner(new_song_df['lyrics'][0]))
        
        new_song_df['lyrics_length'] = [len(lyr)]
        new_song_df['lyrics_num_verbs'] = [len([token.text.strip() for token in lyr if token.pos_=='VERB'])]
        new_song_df['lyrics_num_nouns'] = [len([token.text.strip() for token in lyr if token.pos_=='NOUN'])]
        new_song_df['lyrics_num_adverbs'] = [len([token.text.strip() for token in lyr if token.pos_=='ADV'])]
        new_song_df['lyrics_num_total_punct'] = [len([token.text.strip() for token in lyr if token.pos_ == 'PUNCT'])]
        
        features = new_song_df.columns.drop(str_cat_cols + cat_cols)
        X = new_song_df[['album_release_year'] + list(features)]
        
        #print('Prediction: %s\nProbability: %.2f%%' % (rf.predict(X)[0], np.max(rf.predict_proba(X))*100))
        my_cluster= pd.DataFrame()
        my_cluster['label'] = [rf.predict(X)[0]]
        my_cluster['probability'] = [round(np.max(rf.predict_proba(X))*100, 2)]
    
        a = np.array(X.loc[0])
        aa = a.reshape(1,len(a))
        
        scores = []
        predicted_df = df[df['label']==rf.predict(X)[0]].reset_index(drop=True)
        for i in range(len(predicted_df)):
            b = np.array(predicted_df[X.columns].loc[i])
            ba = b.reshape(1, len(b))
            
            scores.append(np.linalg.norm(aa-ba))
            
        ser_scores = pd.Series(scores)
        my_results = predicted_df[['track_name', 'artists_name']].loc[list(ser_scores.sort_values(ascending = True)[:10].index)]

    return render_template('result.html', results = my_results, cluster = my_cluster)



if __name__ == '__main__':
	app.run(debug=True, use_reloader=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    