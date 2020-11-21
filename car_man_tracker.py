import cv2


video = cv2.VideoCapture('Clueless Pedestrians .mp4')


car_tracker_file = 'cars.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrain_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

while True:
    (read_successful, frame) = video.read()

    if read_successful:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrains = pedestrain_tracker.detectMultiScale(grayscale_frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        if(len(cars)) > 0:
            cv2.putText(frame, 'Car', (x, y+h+40), fontScale=2,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    for (x, y, w, h) in pedestrains:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        if(len(pedestrains)) > 0:
            cv2.putText(frame, 'Pedestrain', (x, y+h+40), fontScale=2,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    cv2.imshow('Car and Pedestrian Tracker', frame)

    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break
