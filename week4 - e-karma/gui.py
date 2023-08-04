import tkinter as tk
import pygame
from PIL import Image, ImageTk
from tkinter import filedialog
from scipy.io.wavfile import write
import numpy as np
from IPython.display import Audio
from pydub import AudioSegment
from pydub.playback import play

from ordering_songs_old import create_mix

"""
import os
directory = "C:/Users/mitta/OneDrive/Documents/BeaverWorksCogWorks"
files = os.listdir(directory)
print(files)

d = "C:/Users/mitta/offspringofpyae-6/week4 - e-karma/test songs"
file = os.listdir(d)
print(file)

pygame.mixer.init()
file = "C:/cartoon.mp3"
pygame.mixer.music.load(file)
pygame.mixer.music.play(loops=0)
"""

global PAUSED
PAUSED = False
global SONG_LIST
SONG_LIST = []

def add_songs():
    global SONG_LIST
    mp3_files_list = filedialog.askopenfilenames(initialdir='./', title="Choose a Song", filetypes=(('mp3 Files', '*.mp3'), ))
    for mp3_file in mp3_files_list:
        window.insert('end', mp3_file)
        SONG_LIST.append(mp3_file)
    print(SONG_LIST)

def play_song():
    mp3_file = window.get('active')
    mp3_file = mp3_file
    pygame.mixer.music.load(mp3_file)
    """
    global SONG_LIST
    print(SONG_LIST)
    mix = create_mix(SONG_LIST)
    print(mix)
    write('playlist.wav', 48000, mix)
    pygame.mixer.music.load('playlist.wav')
    """
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
window = tk.Listbox(root, bg="purple", fg="white", width=60, selectbackground="grey", selectforeground="black")
window.pack(pady=20)

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
song_menu = tk.Menu(root)
root.config(menu=song_menu)
added_menu = tk.Menu(song_menu)
song_menu.add_cascade(label="Add Songs", menu=added_menu)
added_menu.add_command(label="Choose songs to add to the playlist", command=add_songs)

root.mainloop()