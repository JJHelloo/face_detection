import cv2
import os

# Set OpenCV to use a different video I/O backend
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0)
# Set the video resolution to 480p
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Read the frame
    _, img = cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces, rejectLevels, levelWeights = face_cascade.detectMultiScale3(gray, scaleFactor=1.1, minNeighbors=4, outputRejectLevels=True)
    
    # Draw rectangle around the faces and print confidence score
    for (x, y, w, h), weight in zip(faces, levelWeights):
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{weight}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    # Display the output
    cv2.imshow('Face Detection', img)
    
    # Stop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
