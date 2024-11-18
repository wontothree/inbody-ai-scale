import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db, storage

service_account_key_json_path = '/Users/kevinliam/Desktop/anthony/development/inbody-ai-scale/faceRecognition/src/serviceAccountKey.json'

class FireBase():
    def __init__(self):
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_key_json_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': "https://inbody-scale-ai-default-rtdb.firebaseio.com/",
                'storageBucket': "inbody-scale-ai.appspot.com"}
            )
        self.db = db
        self.storage = storage

        self.bucket = self.storage.bucket()

    def upload_data_to_firebase(self, data):
        ref = self.db.reference('Face')
        
        for key, value in data.items():
            ref.child(key).set(value)
        
        print("complete uploading data to firebase")

    def upload_image_to_firebase_storage(self, image_path):
        # Firebase Storage에 이미지 업로드
        if os.path.exists(image_path):  # 경로가 유효한지 확인
            blob = self.bucket.blob(image_path)  # Firebase Storage의 경로
            blob.upload_from_filename(image_path)  # 로컬 파일을 Firebase Storage로 업로드
            print(f"Image {image_path} uploaded successfully to Firebase Storage.")
        else:
            print(f"Error: The file {image_path} does not exist.")

    def upload_inbody_data(self, inbody_data):
        ref = self.db.reference('InBody')
        
        for key, value in inbody_data.items():
            ref.child(key).set(value)
        
        print("complete uploading data to firebase")

    def download_image_from_firebase_storage(self, storage_path, download_path):
        if storage_path:
            # Ensure the local directory exists
            download_dir = os.path.dirname(download_path)
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)  # Create directory if it doesn't exist

            # Correct the path formatting to prevent extra slashes
            storage_path = storage_path.lstrip('/')  # Remove leading slash if present
            
            blob = self.bucket.blob(storage_path)  # Firebase Storage 내의 경로
            try:
                blob.download_to_filename(download_path)  # Firebase Storage에서 로컬 파일로 다운로드
                print(f"Image downloaded to {download_path}")
            except Exception as e:
                print(f"Error downloading image: {e}")
        else:
            print(f"Error: Storage path {storage_path} not found.")

    def get_all_data_from_firebase(self):
        ref = self.db.reference("Face")

        data = ref.get()

        if data:
            print("Data retrival from Firebase completed!")
            return data
        else:
            print("No data found in Firebase!")
            return
        
if __name__ == "__main__":
    firebase = FireBase()

    # print(firebase.get_all_data_from_firebase())

    # firebase.upload_image_to_firebase_storage('search_imgs/tmp.png')

    firebase.download_image_from_firebase_storage(
        'search_imgs/tmp.png',
        'downloaded_imgs/downloaded_tmp.png'
    )