import cv2
import numpy as np
import face_recognition
import pickle

from firebase import FireBase

if __name__ == "__main__":
    firebase = FireBase()

    db, storage = firebase.initializeFirebase()
    bucket = storage.bucket()

    # Set camera capture
    cap = cv2.VideoCapture(0)  # Ensure you are using the correct camera index
    cap.set(3, 1280)  # Width resolution
    cap.set(4, 720)   # Height resolution 

    # locad the encoding file
    print("Loading Encode File ...")
    file = open('/Users/kevinliam/Desktop/anthony/development/inbody-ai-scale/faceRecognition/src/EncodeFile.p', 'rb')
    encodeListWithKnownWithIds = pickle.load(file)
    file.close()
    encodeListKnown, personIds = encodeListWithKnownWithIds
    print("Encode File Loaded")

    modeType = 0
    counter = 0
    id = -1
    imgPerson = []

    while True:
        success, img = cap.read()

        if not success:
            print("Failed to grab frame")
            break

        # make smaller for computation 
        # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25) # reduce size 1/4
        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # imgS

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print("matches", matches)
            # print("faceDis", faceDis)

            matchIndex = np.argmin(faceDis)
            # print("Match Index", matchIndex)

            if matches[matchIndex]:
                # print("Known Face Detected")
                # print(personIds[[matchIndex]])
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = 4 * y1, 4 * x2, 4 * y2, 4 * x1

                id = personIds[matchIndex]
                # print(id)
                if counter == 0:
                    counter = 1
                    modeType = 1

        if counter != 0:
            if counter == 1:
                # get the data from database
                personInfo = db.reference(f'Person/{id}').get()
                print(personInfo)

                # get the image from the storage
                blob = bucket.get_blob(f'imgs/{id}.png')

                if blob is not None:

                    array = np.frombuffer(blob.download_as_string(), np.uint8)  

                    imgPerson = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                # update data of usage
                ref = db.reference(f'Person/{id}')
                personInfo['usage_count'] += 1
                print(personInfo['usage_count'])
                ref.child('usage_count').set(personInfo['usage_count'])
            
            cv2.putText(img, str(personInfo['name']), (861, 125), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1) 
            
            print(personInfo)
            counter += 1

        # Show the combined image
        cv2.imshow("Face Recognition", img) # imgBackground

        # Check for exit key (e.g., 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
