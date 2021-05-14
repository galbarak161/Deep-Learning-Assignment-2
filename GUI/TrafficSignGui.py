import tkinter as tk
from tkinter import filedialog
from tkinter import *

from PIL import ImageTk, Image
import os

from GUI.hw2_labels_dictionary import classes
from Models import model

pathToExamples = os.path.join(os.getcwd(), 'images_examples', 'Meta')

# initialise GUI
top = tk.Tk()
top.geometry('500x400')
top.title('Traffic Sign Classifier')
top.configure(background='#CDCDCD')

signName = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
inputSign = Label(top)
resultsSign = Label(top)


def classify(filePath):
    prediction = model.getPredictions(filePath)
    sign = classes[prediction + 1]

    signName.configure(foreground='#011638', text="Sign: " + sign)

    pathToImage = os.path.join(pathToExamples, f'{prediction}.png')
    signImg = Image.open(pathToImage)

    signImg = signImg.resize((150, 150), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(signImg)
    resultsSign.configure(image=img)
    resultsSign.image = img


def showClassifyButton(filePath):
    classifyButton = Button(top, text="Classify Image", command=lambda: classify(filePath), padx=10, pady=5)
    classifyButton.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classifyButton.place(relx=0.75, rely=0.9, anchor=CENTER)


def uploadImage():
    file_path = filedialog.askopenfilename()
    if file_path == '':
        return
    uploaded = Image.open(file_path)
    uploaded = uploaded.resize((150, 150), Image.ANTIALIAS)
    im = ImageTk.PhotoImage(uploaded)

    inputSign.configure(image=im)
    inputSign.image = im
    signName.configure(text='')
    showClassifyButton(file_path)


heading = Label(top, text="Traffic Sign Recognizer", pady=15, font=('arial', 15, 'bold'))
heading.configure(background='#CDCDCD', foreground='#011638')
heading.place(relx=0.5, rely=0.1, anchor=CENTER)

uploadButton = Button(top, text="Upload an Image", command=uploadImage, padx=10, pady=5)
uploadButton.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
uploadButton.place(relx=0.25, rely=0.9, anchor=CENTER)

inputSign.place(relx=0.25, rely=0.6, anchor=CENTER)
resultsSign.place(relx=0.75, rely=0.6, anchor=CENTER)
signName.place(relx=0.5, rely=0.2, anchor=CENTER)

top.mainloop()
