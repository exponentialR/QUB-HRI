import tkinter as tk
import subprocess
from tkinter import font


def start_label_studio():
    # Use subprocess to call the shell script
    subprocess.Popen(['/home/iamshri/PycharmProjects/QUB-HRI/actionLabelling/label_studio.sh'], shell=True)


def stop_label_studio():
    # Kill all instances of Label Studio
    subprocess.run(['pkill', '-f', 'label-studio'])


# Create the main window
root = tk.Tk()
root.title("Label Studio Manager")
root.geometry('300x200')  # Width x Height
root.config(bg='light blue')  # Background color of the window

# Custom font
customFont = font.Font(family="Helvetica", size=12, weight="bold")

# Add buttons with colors
start_button = tk.Button(root, text="Start Label Studio", command=start_label_studio, bg='green', fg='white',
                         font=customFont)
start_button.pack(pady=20, fill=tk.X, padx=20)

stop_button = tk.Button(root, text="Stop Label Studio", command=stop_label_studio, bg='red', fg='white',
                        font=customFont)
stop_button.pack(pady=20, fill=tk.X, padx=20)

# Run the GUI
root.mainloop()
