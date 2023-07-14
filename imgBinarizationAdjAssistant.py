from CircleDetect import find_circles
import cv2
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

# initialize orig_img as an empty image
orig_img = np.zeros((500, 500, 3), dtype=np.uint8)

def load_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp")])
    global orig_img
    orig_img = cv2.imread(filepath)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    display_image(orig_img)

def display_image(image):
    # calculate the new size to fit the canvas
    ratio = min(canvas.winfo_width() / image.shape[1], canvas.winfo_height() / image.shape[0])
    new_size = (int(image.shape[1] * ratio), int(image.shape[0] * ratio))

    # resize the image
    image = cv2.resize(image, new_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    image = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=NW, image=image)
    canvas.image = image

def adjust_threshold(val):
    if orig_img.size > 0:
        _, binary_img = cv2.threshold(orig_img, int(val), 255, cv2.THRESH_BINARY)
        display_image(binary_img)

def save_image():
    #_, binary_img = cv2.threshold(orig_img, scale.get(), 255, cv2.THRESH_BINARY)
    #cv2.imwrite('result.jpg', binary_img)
    print(scale.get())
    root.destroy()
    

def resize_image(event):
    display_image(orig_img)

root = Tk()

# create a canvas to display the image
canvas = Canvas(root, width=1000, height=1000, bg='white')
canvas.grid(row=0, column=0, sticky=N+S+E+W)

# configure the row and column weights
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

# create a frame for the buttons and scale
frame = Frame(root)
frame.grid(row=1, column=0)

# create a button to load the image
btn_load = Button(frame, text="Load Image", command=load_image)
btn_load.pack(side=LEFT)

# create a scale to adjust the threshold
scale = Scale(frame, from_=0, to=255, orient=HORIZONTAL, command=adjust_threshold)
scale.pack(side=LEFT)

# create a button to save the image
btn_save = Button(frame, text="保存读数", command=save_image)
btn_save.pack(side=LEFT)

# bind the resize event
root.bind('<Configure>', resize_image)

root.mainloop()
