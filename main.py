import os
import argparse

import cv2

from datetime import datetime
import asyncio
import logging

from model.siamese import SiameseModel
from utils.bot import handle_door
from utils.constant import *
from utils.args import args
from utils.firebase import update_door

from tkinter import *
from tkinter import ttk
from tkinter.ttk import Frame

from PIL import Image, ImageTk

camera_num = args['camera']

siamese_model = SiameseModel('model/siamesemodelv2.h5', 0.93, 0.88)

#use with mac cam
#siamese_model = SiameseModel('model/siamesemodelv2.h5', 0.7, 0.65)

cur_attempt = 0
lock_time = 5000

def start_tkinter_app():

    mainWindow = Tk('Verification')
    mainWindow.configure(bg=lightBlue2)
    mainWindow.geometry('%dx%d+%d+%d' % (maxWidth,maxHeight,0,0))
    mainWindow.resizable(0,0)

    # Frame
    mainFrame = Frame(mainWindow)
    mainFrame.place(x=20, y=20)

    #Capture video frames
    lmain = Label(mainFrame)
    lmain.grid(row=0, column=1)

    cap = cv2.VideoCapture(camera_num)

    def show_frame():
        global frame
        _, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        frame = frame[120:120+480, 440:440+480, :]
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cap_img = Image.fromarray(opencv_image)
        photo_image = ImageTk.PhotoImage(image=cap_img)
        lmain.photo_image = photo_image
        lmain.config(image=photo_image)
        lmain.after(20, show_frame)

    def verificate_image():

        cv2.imwrite(os.path.join('data', 'input_data', 'input_img.jpg'), frame)
        r,v = siamese_model.verify()

        global cur_attempt
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if(v):
            cur_attempt = 0
            print('Welcome Home PIKO')
            test.config(text='Welcome Home PIKO')
            update_door(1)
            msg = 'Door Open At' + current_time
            asyncio.run(handle_door(os.path.join('data', 'input_data', 'input_img.jpg'), msg))

        else:
            cur_attempt += 1
            test.config(text=f'Warning : your fail attempt is {cur_attempt} time/times')

            if(cur_attempt == ATTEMPT):
                test.config(text=f'Warning At {cur_attempt} : Your face has been sent to the host.')
                cur_attempt = 0
                update_door(-1)
                msg = f'At {current_time} have someone trying to unlock your door.'
                asyncio.run(handle_door(os.path.join('data', 'input_data', 'input_img.jpg'), msg))



    test = Label(mainWindow, text="Welcome To Our Facial Verification.")
    test.place(x = 150, y = maxHeight-150)

    closeButton = Button(mainWindow, text = "CLOSE", 
    font = fontButtons, bg = white, width = 10, height= 2)
    closeButton.configure(command= lambda: mainWindow.destroy())              
    closeButton.place(x = 50, y = maxHeight-65)

    verificate = Button(mainWindow, text = "VERIFICATE", 
    font = fontButtons, bg = white, width = 10, height= 2)
    verificate.configure(command = verificate_image)              
    verificate.place(x = maxWidth - 170, y = maxHeight-65)

    show_frame()
    mainWindow.mainloop()


if __name__ == "__main__":
    start_tkinter_app()