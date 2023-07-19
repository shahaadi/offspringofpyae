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
from tkinter import ttk


db = Database()
peaks = None
entry_duration = None
label_duration = None
label_song_name = None
label_artist_name = None

def record_microphone():
    global entry_duration, sampling_rate, label_duration, label_song_name, label_artist_name
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
    global label_duration, label_song_name, label_artist_name
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
    file_path = filedialog.asksaveasfilename(initialdir="./", title="Save Database File", filetypes=(("Database Files", "*.pkl"), ("All Files", "*.*")), defaultextension=".pkl")
    db.save_db(db.get_db(), file_path)


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
    song_id = db.query_song(fp.fingerprint(peaks, 15))
    song = db.get_song(song_id)
    if song:
        print("Match found:")
        print("Song Name:", song.name)
        print("Artist Name:", song.artist)
    else:
        print("No match found.")
def visualize_database():
    # Create a new window for visualization
    window = tk.Toplevel()
    window.title("Database Visualization")

    # Create a treeview widget to display the database contents
    tree = ttk.Treeview(window)
    tree["columns"] = ("Name", "Artist", "Duration")

    # Set the column headings
    tree.heading("#0", text="ID")
    tree.heading("Name", text="Song Name")
    tree.heading("Artist", text="Artist Name")
    tree.heading("Duration", text="Duration")

    # Retrieve the database contents
    database = db.get_db()

    # Insert the songs into the treeview
    for song_id, song_data in database.items():
        song_name = entry_song_name.get()
        artist_name = entry_artist_name.get()
        duration = entry_duration.get()
        tree.insert("", "end", text=song_id, values=(song_name, artist_name, duration))

    # Add the treeview to the window
    tree.pack(fill="both", expand=True)

def create_gui(window):
    global entry_duration, entry_song_name, entry_artist_name, ax, canvas, fig, label_duration, label_song_name, entry_song_name

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
    visualize_button = tk.Button(window, text="Visualize Database", command=visualize_database)


    # Create a figure and axis for the spectrogram plot
    fig = plt.Figure(figsize=(8, 4), dpi=80)
    ax = fig.add_subplot(111)

    # Create a canvas to display the matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()

    label_duration.pack()
    entry_duration.pack()
    label_song_name.pack()
    entry_song_name.pack()
    label_artist_name.pack()
    entry_artist_name.pack()

    # Pack the labels, entry fields, and buttons to the window
    microphone_button.pack()
    upload_button.pack()
    load_button.pack()
    save_button.pack()
    add_song_button.pack()
    match_song_button.pack()
    visualize_button.pack()

    canvas.get_tk_widget().pack()
