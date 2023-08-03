import requests, json
import numpy as np
import pandas as pd
from tqdm import tqdm
def getBestTimeStamps(datapoints: np.ndarray, cutoff: float = 0.8):
    new_start = np.argmax((datapoints[1:] - datapoints[:-1]) > 0)
    datapoints[:new_start] = 0
    datapoints *= 1 / np.max(datapoints)

    peak = np.argmax(datapoints)

    good_dp = datapoints > cutoff
    prev_dp = ~good_dp[:peak][::-1]
    if prev_dp.any():
        start_idx = peak - np.argmax(prev_dp)
    else:
        start_idx = 0

    post_dp = ~good_dp[peak:]
    if post_dp.any():
        end_idx = peak + np.argmax(post_dp)
    else:
        end_idx = good_dp.shape[0]

    start_ms = start_idx * t_step
    end_ms = end_idx * t_step

    return (start_ms, end_ms)

song_data = pd.read_csv('video_data.csv')
print(song_data)

for index in tqdm(range(len(song_data['video_id']) - 1, -1, -1)):
    videoID = song_data.at[index, 'video_id']
    if song_data.loc[[index]].isna().sum().sum() > 1:
        # get video activity from video
        url = f'https://yt.lemnoslife.com/videos?part=mostReplayed&id={videoID}'
        content = requests.get(url).text
        data = json.loads(content)
        datapoints = np.empty((100,))
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
                    heatMarker = heatMarker['heatMarkerRenderer']
                    intensityScoreNormalized = heatMarker['heatMarkerIntensityScoreNormalized']
                    datapoints[i] = intensityScoreNormalized
                song_data.at[index, 'optimal_start'], song_data.at[index, 'optimal_end'] = getBestTimeStamps(datapoints)
                song_data.at[index, 'song_duration'] = duration

                # get video_title
                url = f'https://yt.lemnoslife.com/noKey/videos?part=snippet&id={videoID}'
                content = requests.get(url).text
                data = json.loads(content)
                song_data.at[index, 'video_title'] = data['items'][0]['snippet']['title']

                # get song_name
                url = f'https://yt.lemnoslife.com/videos?part=musics&id={videoID}'
                content = requests.get(url).text
                data = json.loads(content)
                if len(data['items'][0]['musics']) != 0:
                    song_data.at[index, 'song_name'] = data['items'][0]['musics'][0]['song']['title']

                # get viewcount of video
                url = f'https://yt.lemnoslife.com/noKey/videos?part=statistics&id={videoID}'
                content = requests.get(url).text
                data = json.loads(content)
                song_data.at[index, 'viewcount'] = data['items'][0]['statistics']['viewCount']
        if index % 100 == 0:
            song_data.to_csv('video_data.csv')