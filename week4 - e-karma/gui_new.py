import tkinter as tk
import pygame
from PIL import Image, ImageTk
from tkinter import filedialog
from scipy.io.wavfile import write
import numpy as np
from IPython.display import Audio
from pydub import AudioSegment
from pydub.playback import play
from urllib.parse import urlparse
import tkinter.ttk as ttk
import tensorflow as tf

from ordering_songs import create_mix
from Spotify_Youtube_API import get_songs_artists, get_token, find_videoID
from spectrogram_download import video_ids_spectrograms

model = tf.keras.models.load_model('week4 - e-karma/test1.keras')

global PAUSED
PAUSED = False
global MIX
MIX = None

def prediction_to_index(pred, cutoff):
    pred *= 1 / pred.max()
    peak = np.argmax(pred)
    good_dp = pred > cutoff
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

    dif = end_idx - start_idx
    if dif < 3:
        if (start_idx + 3) > good_dp.shape[0]:
            start_idx = end_idx - 3
        else:
            end_idx = start_idx + 3

    return start_idx, end_idx

def add_songs():
    # spotify_playlist = text_box.get("1.0", "end-1c")
    spotify_playlist = 'https://open.spotify.com/playlist/7IXaLrFAFxmUELfKUycf1H?si=5b9991759b0d479b'
    spotify_string = urlparse(spotify_playlist).path.split('/')[-1]
    print(spotify_string)
    l = get_songs_artists(get_token(), spotify_string)
    l = l[0:20]
    video_ids = find_videoID(l)
    audio, spectrograms = video_ids_spectrograms(video_ids)
    spectrograms = np.expand_dims(spectrograms, axis=-1)
    pred = model.predict(spectrograms)
    
    cutoff = 0.8
    
    audio_files = []
    for i in range(len(video_ids)):
        start_idx, end_idx = prediction_to_index(pred[i], cutoff)
        audio_files.append(audio[i, start_idx * 48000:end_idx * 48000])
    
    global MIX
    MIX = create_mix(audio_files) # pass in spotify playlist to code for determining artist and song names, then pass that to model, then pass to order, then get the mix
    write('playlist.wav', 48000, MIX)
    pygame.mixer.music.load('playlist.wav')
    pygame.mixer.music.play(loops=0)

def play_song():
    # mp3_file = window.get('active')
    # mp3_file = mp3_file
    # pygame.mixer.music.load(mp3_file)
    global MIX
    write('playlist.wav', 48000, MIX)
    pygame.mixer.music.load('playlist.wav')
    pygame.mixer.music.play(loops=0)
    
def stop_song():
    PAUSED = False
    pygame.mixer.music.stop()
    
def pause_song(paused):
    global PAUSED
    PAUSED = paused
    if not PAUSED:
        pygame.mixer.music.pause()
        PAUSED = True
    else:
        pygame.mixer.music.unpause()
        PAUSED = False
    
    

root = tk.Tk()
root.title("AI Music Mixer")
root.geometry("500x300")

pygame.mixer.init()

"""
window = tk.Listbox(root, bg="purple", fg="white", width=60, selectbackground="orange", selectforeground="grey")
window.pack(pady=20)
"""

text_box = tk.Text(root, height=10, width=40)
text_box.pack()
open_btn = tk.Button(root, text="Link to Spotify Playlist", command=add_songs)
open_btn.pack()

# create buttons
back_btn_img = Image.open('./week4 - e-karma/gui_pictures/previous_song_button_2.png')
back_btn_img = back_btn_img.resize((50, 50))
back_btn_img = ImageTk.PhotoImage(back_btn_img)

forward_btn_img = Image.open('./week4 - e-karma/gui_pictures/next_song_button_2.png')
forward_btn_img = forward_btn_img.resize((70, 70))
forward_btn_img = ImageTk.PhotoImage(forward_btn_img)

play_btn_img = Image.open('./week4 - e-karma/gui_pictures/play_button_2.png')
play_btn_img = play_btn_img.resize((50, 50))
play_btn_img = ImageTk.PhotoImage(play_btn_img)

pause_btn_img = Image.open('./week4 - e-karma/gui_pictures/pause_button.png')
pause_btn_img = pause_btn_img.resize((50, 50))
pause_btn_img = ImageTk.PhotoImage(pause_btn_img)

stop_btn_img = Image.open('./week4 - e-karma/gui_pictures/stop_button.png')
stop_btn_img = stop_btn_img.resize((50, 50))
stop_btn_img = ImageTk.PhotoImage(stop_btn_img)

# create button functionality
frame = tk.Frame(root)
frame.pack()


back_btn = tk.Button(frame, image=back_btn_img, borderwidth=0)
forward_btn = tk.Button(frame, image=forward_btn_img, borderwidth=0)
play_btn = tk.Button(frame, image=play_btn_img, borderwidth=0, command=play_song)
pause_btn = tk.Button(frame, image=pause_btn_img, borderwidth=0, command=lambda: pause_song(PAUSED))
stop_btn = tk.Button(frame, image=stop_btn_img, borderwidth=0, command=stop_song)

back_btn.grid(row=0, column=0, padx=10)
forward_btn.grid(row=0, column=4, padx=0)
play_btn.grid(row=0, column=2, padx=10)
pause_btn.grid(row=0, column=1, padx=10)
stop_btn.grid(row=0, column=3, padx=10)


# create the box for inputting songs
"""
song_menu = tk.Menu(root)
root.config(menu=song_menu)
added_menu = tk.Menu(song_menu)
song_menu.add_cascade(label="Add Songs", menu=added_menu)
added_menu.add_command(label="Choose songs to add to the playlist", command=add_songs)
"""

root.mainloop()