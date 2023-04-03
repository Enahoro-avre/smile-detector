import cv2

face_classifier = 'haarcascade_frontalface_default.xml'
smile_classifier = 'haarcascade_smile.xml'

face_detector = cv2.CascadeClassifier(face_classifier)
smile_detector = cv2.CascadeClassifier(smile_classifier)

# Grab webcam stream
webcam = cv2.VideoCapture(0)

while True:
    #Read the current frame from the webcam
    successful_frame_read , frame = webcam.read()

    if not successful_frame_read:
        print('NOT SUCCESSFUL')
        break
    grayscaled_frame = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(grayscaled_frame)
    
    for ( x , y , w , h ) in faces:
      # Draw a rectangle on th face  
      cv2.rectangle(frame , ( x , y ), (x+w , y+h) , ( 100 , 200 , 50 ), 4 )

        # get the sub-frame using numpy
      the_face = frame[y:y+h , x:x+w]

      # Change to grayscale
      face_grayscale = cv2.cvtColor(the_face , cv2.COLOR_BGR2GRAY) 

      smile = smile_detector.detectMultiScale(face_grayscale , scaleFactor = 1.7 , minNeighbors = 20)

    #   for ( x_ , y_ , w_ , h_ ) in smile:
    #   # Draw a rectangle on the face  
    #     cv2.rectangle(the_face , ( x_ , y_ ), (x_ + w_ , y_ + h_) , ( 50 , 50 , 200 ), 4 )
      if len(smile) > 0:
            cv2.putText(frame , 'Smiling' , (x , y+h+40) , fontScale=3 , 
            fontFace=cv2.FONT_HERSHEY_PLAIN , color = (255 , 255 , 255))
  
    # Show the current frame from the webcam
    cv2.imshow('Smile Detector' , frame)
    key = cv2.waitKey(1)

    # Press Q to break the video frame
    if key == 81 or key == 113:
      break

# Clean up code
webcam.release()





print('Code Completed')