import skimage.io as io
from database import Database

from model import Model
from model import display_faces
import tkinter as tk
from tkinter import filedialog
import gui
import sys

class FaceRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition App")
        self.database = gui.load_database(input("Enter the database filename (e.g., something.pkl): "))
        self.additions_made = False  # Flag to track if additions were made to the database
        self.create_widgets()

    def create_widgets(self):
        self.label_filename = tk.Label(self, text="No image selected.")
        self.label_filename.pack()

        self.select_button = tk.Button(self, text="Select Image", command=self.select_image)
        self.select_button.pack()

        self.recognize_button = tk.Button(self, text="Recognize Faces", command=self.recognize_faces)
        self.recognize_button.pack()

        self.add_to_db_button = tk.Button(self, text="Add to Database", command=self.add_to_database)
        self.add_to_db_button.pack()

        self.quit_button = tk.Button(self, text="Quit", command=self.quit)
        self.quit_button.pack()

    def select_image(self):
        if sys.platform == 'win32':
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        else:
            file_path = filedialog.askopenfilename()

        if file_path:
            self.image_path = file_path
            self.label_filename.config(text=f"Selected Image: {file_path}")
        else:
            self.label_filename.config(text="No image selected.")

    def recognize_faces(self):
        if hasattr(self, "image_path"):
            self.database.display_database()
            recognized_faces, valid_descriptors = gui.recognize_and_display_faces(self.image_path, self.database)
            self.valid_descriptors = valid_descriptors

    def add_to_database(self):
        if hasattr(self, "valid_descriptors"):
            name = input("Enter name of person in image: ")
            gui.add_to_database(name, self.valid_descriptors[0], self.database)
            self.additions_made = True  # Set the flag to True
            print(f"{name} has been added to the database.")

    def quit(self):
        # Save the database and close the GUI if additions have been made
        if self.additions_made:
            database_filename = input("Enter the database filename (e.g., something.pkl): ")
            if database_filename:
                self.database.save_db(database_filename)
                print("Database saved.")
        super().quit()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()
