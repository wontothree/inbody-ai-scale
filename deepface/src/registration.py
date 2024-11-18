import time
from datetime import datetime
import uuid

from capture import Capture
from firebase import FireBase
from faceRecognition import FaceRecognition

class Registration:
    def __init__(self):
        self.data = {}

    def input_user_info(self):
        user_uuid = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        name = input("Name: ")
        sex = input("Sex: ")
        age = input("Age: ")
        
        return user_uuid, timestamp, name, sex, age
    
    def save_user_info(self, user_uuid, timestamp, name, sex, age, embedding):
        self.data[user_uuid] = {
            "time": timestamp,
            "name": name,
            "sex": sex,
            "age": age,
            "embedding": embedding,
        }

        print("complete saving user information")

if __name__ == "__main__":
    capture = Capture()
    face_registration = Registration()
    firebase = FireBase()
    face_recognition = FaceRecognition()

    # input user information
    user_uuid, timestamp, name, sex, age = face_registration.input_user_info()

    img_path = f'./registration_imgs/{user_uuid}.png'

    # capture face image
    capture.capture_photo(img_path)

    time.sleep(3)

    # generate embedding
    embedding = face_recognition.generate_embedding(img_path)

    # save user information
    face_registration.save_user_info(user_uuid, timestamp, name, sex, age, embedding)

    # upload data to firebase 
    firebase.upload_data_to_firebase(face_registration.data)

    # upload image to firebase
    # firebase.upload_image_to_firebase_storage(img_path)
