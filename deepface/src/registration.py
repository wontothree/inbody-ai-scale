import time

from capture import Capture
from firebase import FireBase
from faceRecognition import FaceRecognition

class Registration:
    def __init__(self):
        self.data = {}

    def input_user_info(self):
        index = input("Index: ")
        name = input("Name: ")
        sex = input("Sex: ")
        age = input("Age: ")
        
        return index, name, sex, age
    
    def save_user_info(self, index, name, sex, age, embedding):
        # 사용자 정보를 딕셔너리에 저장
        self.data[index] = {
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
    index, name, sex, age = face_registration.input_user_info()

    # capture face image
    capture.capture_photo(f'./deepface/registration_imgs/{index}.png')

    time.sleep(4)

    # generate embedding
    embedding = face_recognition.generate_embedding(f'./deepface/registration_imgs/{index}.png')

    # save user information
    face_registration.save_user_info(index, name, sex, age, embedding)

    # upload to firebase
    firebase.upload_data_to_firebase(face_registration.data)
