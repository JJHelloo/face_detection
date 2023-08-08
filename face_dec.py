import cv2
import matplotlib.pyplot as plt

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Create a figure and its axes
fig, ax = plt.subplots()

while True:
    # Read the frame
    _, img = cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the output using matplotlib
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(0.01)
    
    # Clear the axes for the next frame
    ax.clear()

# Release the VideoCapture object
cap.release()
