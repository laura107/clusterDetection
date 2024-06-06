import tkinter as tk
from tkinter import Button, Frame, Label
import cv2
import numpy as np
from PIL import Image, ImageTk
import pandas as pd

# variable to store captured image
captured_image = None

def detect_clusters(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale

    # Gaussian blur to reduce noise
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # edge detection using Canny
    edges = cv2.Canny(img_blur, 30, 100)  # Adjust the thresholds as needed

    # find contours in edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()

    for contour in contours:
        # compute the area of the contour
        area = cv2.contourArea(contour)

        # remove small noise
        if area < 100:  # adjust area threshold
            continue

        # approximate contour to a closed polygon
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)  # adjust the epsilon parameter

        # compute circularity of contour
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # refine contours based on circularity
        if circularity < 0.7:  # adjustable circularity threshold
            continue

        # compute bounding rectangle of contour
        x, y, w, h = cv2.boundingRect(contour)

        # compute aspect ratio of the bounding rectangle
        aspect_ratio = w / float(h)

        # focus on more rounded shapes
        if aspect_ratio < 0.5:  # adjust aspect ratio threshold
            continue

        # fit circle to the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # draw the circle on output image
        cv2.circle(output, center, radius, (0, 0, 255), 2) 

    return output


def update_frame():
    if not viewing_captured_image:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # mirror image

            # process frame to detect clusters
            processed_frame = detect_clusters(frame)

            # convert processed frame to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # convert processed frame to PIL format
            img = Image.fromarray(processed_frame_rgb)
            img = ImageTk.PhotoImage(img)

            # update label with processed frame
            label.config(image=img)
            label.image = img

    # call this function again after 10 ms
    root.after(10, update_frame)

def capture_image():
    global captured_image
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1) 
        captured_image = detect_clusters(frame)
        
        # display captured image with circles around clusters
        img = Image.fromarray(cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(img)
        
        label.config(image=img)
        label.image = img

def show_live_view():
    global viewing_captured_image
    viewing_captured_image = False

def show_captured_view():
    global viewing_captured_image
    if captured_image is not None:
        viewing_captured_image = True
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)))
        label.config(image=img)
        label.image = img

# initialize camera
cap = cv2.VideoCapture(0)
viewing_captured_image = False

# create main window
root = tk.Tk()
root.title("Camera Capture with Tkinter")

# create frame for buttons
button_frame = Frame(root, padx=10, pady=10)
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

# create frame for camera feed
image_frame = Frame(root, padx=10, pady=10)
image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# create a label to display image
label = Label(image_frame)
label.pack()

# buttons to capture image and toggle live view
capture_button = Button(button_frame, text="Capture Image", command=capture_image, bg="green", fg="white", padx=10, pady=5)
capture_button.grid(row=0, column=0, padx=5, pady=5)

live_view_button = Button(button_frame, text="Live View", command=show_live_view, bg="blue", fg="white", padx=10, pady=5)
live_view_button.grid(row=0, column=1, padx=5, pady=5)

captured_view_button = Button(button_frame, text="Captured View", command=show_captured_view, bg="orange", fg="white", padx=10, pady=5)
captured_view_button.grid(row=0, column=2, padx=5, pady=5)

update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
