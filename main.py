import cv2
import pytesseract

# Load the video
cap = cv2.VideoCapture('car_video.mp4')

# Load the license plate cascade classifier
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    if ret:
        # Preprocess the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Detect the license plates
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        # Extract the characters and recognize the license plates
        for (x, y, w, h) in plates:
            plate = gray[y:y+h, x:x+w]
            plate = cv2.bitwise_not(plate)
            text = pytesseract.image_to_string(plate, config='--psm 11')
            
            # Draw a bounding box around the license plate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Put the recognized text on the frame
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('frame', frame)
        
        # Exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
