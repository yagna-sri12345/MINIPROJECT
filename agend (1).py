import cv2

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    
    # Preprocess the frame for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    
    # Perform face detection
    detection = faceNet.forward()
    
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            # Extract the bounding box coordinates
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            
            # Add the bounding box coordinates to the list
            bboxs.append([x1, y1, x2, y2])
            
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Return the modified frame and the list of bounding boxes
    return frame, bboxs

# Path to face detection model files
faceProto = "C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\opencv_face_detector.pbtxt"
faceModel = "C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\opencv_face_detector_uint8.pb"

# Path to age prediction model files
ageProto = "C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\age_deploy.prototxt"
ageModel = "C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\age_net.caffemodel"

# Path to gender prediction model files
genderProto = "C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\gender_deploy.prototxt"
genderModel = "C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\gender_net.caffemodel"

# Load face detection model
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Load age prediction model
ageNet = cv2.dnn.readNet(ageModel, ageProto)

# Load gender prediction model
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define the labels for age and gender categories
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Path to advertisement videos based on age and gender
advertisementVideos = {
    'Male': {    
        '(0-2)'  :'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\4.mp4',
        '(4-6)'  :'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\4.mp4',
        '(8-12)' :'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\4.mp4',
        '(15-20)':'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\4.mp4',
        '(25-32)':'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\4.mp4',
        '(38-43)':'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\4.mp4',
        '(48-53)':'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\4.mp4',
        '(60-100)':'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\4.mp4',
    },
    'Female': {
         '(0-2)'  :'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\3.mp4',
        '(4-6)'  :'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\3.mp4',
        '(8-12)' :'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\3.mp4',
        '(15-20)':'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\3.mp4',
        '(25-32)':'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\3.mp4',
        '(38-43)':'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\3.mp4',
        '(48-53)':'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\3.mp4',
        '(60-100)':'C:\\Users\\MEGHANA KARANAM\\Desktop\\mini project\\Age _ Gender Recongniition\\3.mp4'
    }
}

# Open webcam
video = cv2.VideoCapture(0)

padding = 20

while True:
    # Read a frame from the webcam
    ret, frame = video.read()
    
    # Perform face detection and get the modified frame and bounding boxes
    frame, bboxs = faceBox(faceNet, frame)
    
    # Iterate through the detected faces and display advertisements
    for bbox in bboxs:
        # Extract the face region
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                     max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        
        # Preprocess the face region for gender prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Perform gender prediction
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]
        
        # Preprocess the face region for age prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        
        # Perform age prediction
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]
        
        # Construct the label with gender and age
        label = "{},{}".format(gender, age)
        
        # Draw a rectangle and label on the frame
        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1) 
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Get the corresponding advertisement video path based on gender and age
        advertisementPath = advertisementVideos.get(gender, {}).get(age)
        
        if advertisementPath:
            # Open the advertisement video
            advertisement = cv2.VideoCapture(advertisementPath)
            
            while True:
                # Read a frame from the advertisement video
                ret_ad, frame_ad = advertisement.read()
                
                if not ret_ad:
                    # If the advertisement video ends, break the loop
                    break
                
                # Display the advertisement frame
                cv2.imshow("Advertisement", frame_ad)
                
                # Check for key press event
                key = cv2.waitKey(1)
                
                if key == ord('q'):
                    # If 'q' is pressed, stop the advertisement video and exit the loop
                    advertisement.release()
                    break
        else:
            # If no advertisement video found, display a default message
            cv2.putText(frame, "No Advertisement Available", (bbox[0], bbox[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display the frame with detected faces and advertisements
    cv2.imshow("Age-Gender Detection", frame)
    
    # Check for key press event
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        # If 'q' is pressed, exit the loop
        break

# Release the webcam and close all windows
video.release()
cv2.destroyAllWindows()
