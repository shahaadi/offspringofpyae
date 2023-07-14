import get_samples as sample
import fingerprint as fp
import find_peaks as peak
import database as db
import tkinter as tk
import gui as gui

# samples, sampling_rate = sample.micInput(5)
# peaks = peak.spectrogram_plot(samples, sampling_rate)
# fingerprints = fp.fingerprint(peaks, 15)




# db = db.Database()
# song1 = Metadata()
# song2 = Metadata()
# song3 = Metadata()

window = tk.Tk()
window.title("Audio Capstone Project")
gui.create_gui(window)
window.mainloop()
