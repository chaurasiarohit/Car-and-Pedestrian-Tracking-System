import cv2

# Our Video 
video = cv2.VideoCapture('Track1min.mp4')

# Our Pre-trained Classifier
car_tracker_file = 'Car_detector.xml'
pedestrian_tracker_file = 'Pedestrian_detector.xml'

#Create Car and Pedestrian Classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


#Run forever until car stops or something
while True:
    #Read the current frame
    read_successful, frame = video.read()
    if read_successful:
        #Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    #Detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    #Detect Pedestrians
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #Draw rectangles around the cars
    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #Draw rectangles around the pedestrians
    for(x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    #Display
    cv2.imshow('Car and Pedestrian Tracking', frame)

    #Dont autoclose (wait for a press key)
    key = cv2.waitKey(1)

    #Stop if q key is pressed
    if key==81 or key==113:
        break

#Release the videocapture object
video.release()

print("Code Completed")



