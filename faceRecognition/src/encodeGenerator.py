import os
import cv2
import face_recognition
import pickle

class EncodeGenerator:
    def __init__(self):
        # Firebase 초기화
        self.bucket = self.initialize_firebase()

    def initialize_firebase(self):
        firebase = FireBase()
        _, storage = firebase.initializeFirebase()
        return storage.bucket()
    
    def find_encodings(self, images_list):
        encode_list = []
        for img in images_list:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encode_list.append(encode)
        return encode_list
    
    def load_images_from_folder(self, folder_path):
        path_list = os.listdir(folder_path)
        img_list = []
        person_ids = []
        for path in path_list:
            img_list.append(cv2.imread(os.path.join(folder_path, path)))
            person_ids.append(os.path.splitext(path)[0])
        return img_list, person_ids
    
    def upload_images_to_firebase(self, folder_path):
        path_list = os.listdir(folder_path)
        for path in path_list:
            file_name = os.path.join(folder_path, path)
            blob = self.bucket.blob(file_name)
            blob.upload_from_filename(file_name)

    def save_encodings_to_file(self, encode_list_with_ids, file_name="./EncodeFile.p"):
        with open(file_name, 'wb') as file:
            pickle.dump(encode_list_with_ids, file)
        print("File Saved")

if __name__ == "__main__":
    encodeGenerator = EncodeGenerator()
    
    # 로컬 폴더에서 이미지와 ID 불러오기
    folder_path = '/Users/kevinliam/Desktop/anthony/development/inbody-ai-scale/faceRecognition/imgs'
    img_list, person_ids = encodeGenerator.load_images_from_folder(folder_path)

    # Firebase 스토리지에 이미지 업로드
    encodeGenerator.upload_images_to_firebase(folder_path)

    # 얼굴 인코딩 생성
    print("Encoding Started ...")
    encode_list_known = encodeGenerator.find_encodings(img_list)
    encode_list_known_with_ids = [encode_list_known, person_ids]
    print("Encoding Complete")

    # 인코딩과 ID를 파일로 저장
    encodeGenerator.save_encodings_to_file(encode_list_known_with_ids)
    