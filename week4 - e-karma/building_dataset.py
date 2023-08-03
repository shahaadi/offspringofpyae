import numpy as np
import pandas as pd
import requests, json
from yt_dlp import YoutubeDL
from tqdm import tqdm
import os

# pulls song ids and titles
def save_pre_data(channel_id, path):
    SEP = ';'

    try:
        os.remove(path)
    except OSError:
        pass

    basecommand = f'python -m yt_dlp --flat-playlist -j --print-to-file "{channel_id};%(id)s;%(title)s;;" "{path}" "https://www.youtube.com/channel/{channel_id}"'
    os.system(basecommand)

    song_data = pd.read_csv(path, sep=SEP, header=None)
    columns = ['channel_id', 'video_id', 'title', 'duration', 'view_count']
    song_data.columns = columns

    song_data[[f'datapoint_{i}' for i in range(100)]] = np.nan
    song_data.to_csv(path, sep=SEP)

# pull song data and save to path
def save_song_data(path):
    SEP = ';'
    song_data = pd.read_csv(path, sep=SEP)

    for index in tqdm(range(len(song_data['video_id']) - 1, -1, -1)):
        videoID = song_data.at[index, 'video_id']
        if song_data.loc[[index]].isna().sum().sum() != 0:
            # get video activity from video
            url = f'https://yt.lemnoslife.com/videos?part=mostReplayed&id={videoID}'
            content = requests.get(url).text
            data = json.loads(content)
            print(videoID)
            if data['items'][0]['mostReplayed'] is None:
                song_data = song_data.drop(index)
            else:
                # get song_duration
                url = f'https://yt.lemnoslife.com/videos?part=contentDetails&id={videoID}'
                content = requests.get(url).text
                time_data = json.loads(content)
                duration = time_data['items'][0]['contentDetails']['duration']
                if duration > 600:
                    song_data = song_data.drop(index)
                else:
                    t_step = data['items'][0]['mostReplayed']['heatMarkers'][0]['heatMarkerRenderer']['markerDurationMillis']
                    for i, heatMarker in enumerate(data['items'][0]['mostReplayed']['heatMarkers']):
                        song_data.at[index, f'datapoint_{i}'] = heatMarker['heatMarkerRenderer']['heatMarkerIntensityScoreNormalized']
                    song_data.at[index, 'duration'] = duration

                    # get view_count of video
                    url = f'https://yt.lemnoslife.com/noKey/videos?part=statistics&id={videoID}'
                    content = requests.get(url).text
                    data = json.loads(content)
                    song_data.at[index, 'view_count'] = data['items'][0]['statistics']['viewCount']
                    song_data.to_csv(path, sep=SEP)

# removes leading columns that are not channel_id
def remove_irrev_cols(path):
    song_data = pd.read_csv(path)
    while song_data.columns[0] != 'channel_id':
        song_data = song_data.drop(song_data.columns[0], axis=1)

# cleans data by removing non songs
def remove_non_songs(channel_id, path):
    keyword = None
    if channel_id == ''
    if keyword is not None:
        for index in range(len(song_data['video_id']) - 1, -1, -1):


# all in one function
def save_data(channel_id):
    save_pre_data(channel_id, f'{channel_id}_output.csv')
    save_song_data(f'{channel_id}_output.csv')
    remove_irrev_cols(f'{channel_id}_output.csv')
    remove_non_songs(channel_id, f'{channel_id}_output.csv')

# merge all channel csvs
def save_and_concat(channel_ids):
    dfs = []
    for channel_id in channel_ids:
        save_data(channel_id)
        dfs.append(pd.read_csv(f'{channel_id}_output.csv', sep=';'))
    df = pd.concat(dfs)
    df.to_csv('output.csv', sep=';')
