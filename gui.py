import tkinter as tk
from tkinter import filedialog
import numpy as np
import librosa
import get_samples as sample
import fingerprint as fp
import find_peaks as peak
import database as db
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from database import Database
from metadata import MetaData

db = Database()

def record_microphone():
    global entry_duration, sampling_rate
    duration = entry_duration.get()
    if duration:
        try:
            duration = float(duration)
            samples, sampling_rate = sample.micInput(duration)
            update_spectrogram(samples, sampling_rate)
        except ValueError:
            # Handle the case when the input is not a valid float
            print("Invalid duration. Please enter a valid numeric value.")
    else:
        # Handle the case when the input is empty
        print("Please enter a duration value.")

def upload_file():
    file_path = filedialog.askopenfilename(initialdir="./", title="Select Recording File", filetypes=(("Audio Files", "*.wav"), ("Audio Files", "*.mp3"), ("All Files", "*.*")))
    samples, sampling_rate = sample.fileInput(file_path)
    update_spectrogram(samples, sampling_rate)

def update_spectrogram(samples, sampling_rate):
    global peaks, ax, fig, samples_len
    peaks, fig, ax = peak.spectrogram_plot(samples, sampling_rate)
    samples_len = samples.shape[0]

    canvas.figure = fig
    canvas.draw()

def load_database():
    file_path = filedialog.askopenfilename(initialdir="./", title="Select Database File", filetypes=(("Database Files", "*.pkl"), ("All Files", "*.*")))
    db.load_db(file_path)

def save_database():
    file_path = filedialog.askopenfilename(initialdir="./", title="Select Database File", filetypes=(("Database Files", "*.pkl"), ("All Files", "*.*")))
    db.save_db(file_path)

def add_song():
    # Code to add a song to the current database
    global entry_song_name, entry_artist_name, peaks, samples_len

    song_name = entry_song_name.get()
    artist_name = entry_artist_name.get()
    fingerprint = fp.fingerprint(peaks, 15)
    song_data = MetaData(song_name, artist_name, samples_len)
    db.add_song(song_data, fingerprint)

def match_song():
    # Code to match a song
    global peaks
    song_id = db.query_song(fingerprint(peaks, 15))
    print(MetaData.getSong(song_id).name)

def create_gui(window):
    global entry_duration, entry_song_name, entry_artist_name, ax, canvas, fig

    # Create labels and entry fields for parameters
    label_duration = tk.Label(window, text="Duration (seconds):")
    entry_duration = tk.Entry(window)
    label_song_name = tk.Label(window, text="Song Name")
    entry_song_name = tk.Entry(window)
    label_artist_name = tk.Label(window, text="Artist Name")
    entry_artist_name = tk.Entry(window)

    # Create buttons for microphone recording and file upload
    microphone_button = tk.Button(window, text="Record from Microphone", command=record_microphone)
    upload_button = tk.Button(window, text="Upload Recording File", command=upload_file)

    load_button = tk.Button(window, text="Load Database", command=load_database)
    save_button = tk.Button(window, text="Save Database", command=save_database)
    add_song_button = tk.Button(window, text="Add Song to Database", command=add_song)
    match_song_button = tk.Button(window, text="Match Song", command=match_song)


    # Create a figure and axis for the spectrogram plot
    fig = plt.Figure(figsize=(8, 4), dpi=80)
    ax = fig.add_subplot(111)

    # Create a canvas to display the matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()

    # Pack the labels, entry fields, and buttons to the window
    label_duration.pack()
    entry_duration.pack()
    label_song_name.pack()
    entry_song_name.pack()
    label_artist_name.pack()
    entry_artist_name.pack()
    microphone_button.pack()
    upload_button.pack()
    load_button.pack()
    save_button.pack()
    add_song_button.pack()
    match_song_button.pack()
    canvas.get_tk_widget().pack()
