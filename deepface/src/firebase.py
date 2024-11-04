import firebase_admin
from firebase_admin import credentials
from firebase_admin import db, storage

class FireBase():
    def __init__(self):
        if not firebase_admin._apps:
            cred = credentials.Certificate("/Users/kevinliam/Desktop/anthony/development/inbody-ai-scale/faceRecognition/src/serviceAccountKey.json")
            firebase_admin.initialize_app(cred, {
                'databaseURL': "https://inbody-scale-ai-default-rtdb.firebaseio.com/",
                'storageBucket': "inbody-scale-ai.appspot.com",
            })
        self.db = db
        self.storage = storage

    def upload_data_to_firebase(self, data):
        ref = self.db.reference('Person')
        
        for key, value in data.items():
            ref.child(key).set(value)
        
        print("complete uploading data to firebase")

    def get_all_data_from_firebase(self):
        ref = self.db.reference("Person")

        data = ref.get()

        if data:
            print("Data retrival from Firebase completed!")
            return data
        else:
            print("No data found in Firebase!")
            return
        
if __name__ == "__main__":
    firebase = FireBase()

    print(firebase.get_all_data_from_firebase())