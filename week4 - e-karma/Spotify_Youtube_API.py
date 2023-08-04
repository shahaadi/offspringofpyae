import base64
from requests import post, get
import json

def get_token():
    CLIENT_ID = "6f9bd56c69774d6abaa2ca927b7b5da8"
    CLIENT_SECRET = "b1c40b64d8764ac28727eec418d969d9"
    auth_string = CLIENT_ID + ":" + CLIENT_SECRET
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded" 
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers,data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

def get_songs_artists(token, playlist_id):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = get_auth_header(token)
    result = get(url, headers = headers)
    json_result = json.loads(result.content)
    data = []
    if "tracks" in json_result:
        json_result = json_result["tracks"]
    for i in range(len(json_result["items"])):
        song = json_result["items"][i]["track"]
        song_name = song["name"]
        artist_names = []
        for artist in song["artists"]:
            if len(song["artists"])>1:
                artist_names.append(artist["name"])
            else:
                artist_names = artist["name"]
        data.append((song_name,artist_names))
    return data

def get_playlists(token, user_id):
    url = f"https://api.spotify.com/v1/users/{user_id}/playlists"
    
    headers = get_auth_header(token)
    result = get(url, headers = headers)
    json_result = json.loads(result.content)
    playlists = json_result["items"]
    print("Choose playlist")
    for i in range(len(playlists)):
        print(f"({i+1}) " + playlists[i]["name"])

    i = int(input())
    assert 1 <= i <= len(playlists)
    i-= 1
    playlist_id = playlists[i]["id"]
    return get_songs_artists(token, playlist_id)



from apiclient.discovery import build

def find_videoID(l):
    yt_api_key = "AIzaSyB9eZn4K3wfYAJU4mfovsz0Z2q5KoWsQx4"
    videoIds = []
    titles = []
    for i in range(len(l)):
        youtube = build("youtube", "v3", developerKey = yt_api_key)
        song, artist = l[i]
        req = youtube.search().list(q= song + " " + " ".join(artist) + " lyrics", part = "snippet", type="video")
        res = req.execute()
        videoIds.append(res["items"][0]["id"]["videoId"])
        titles.append(res["items"][0]["snippet"]["title"])
    return videoIds, titles
