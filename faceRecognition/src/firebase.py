import firebase_admin
from firebase_admin import credentials
from firebase_admin import db, storage

data = {
    "111111" :
        {
             "name" : "Sewon Kim",
             "age" : 23,
             "last_using_time" : "2024-09-01 00:00:01",
             "usage_count" : 10,
        },
    "222222" :
        {
             "name" : "Elon Musk",
             "age" : 53,
             "last_using_time" : "2024-08-31 00:00:01",
             "usage_count" : 12,
        },
    "111112" :
        {
             "name" : "김영진",
             "age" : 24,
             "last_using_time" : "2024-09-01 00:00:01",
             "usage_count" : 15,
        },
    "111113" :
        {
             "name" : "박승서",
             "age" : 21,
             "last_using_time" : "2024-09-01 00:00:01",
             "usage_count" : 15,
        },
    "111114" :
        {
             "name" : "설민관",
             "age" : 21,
             "last_using_time" : "2024-09-01 00:00:01",
             "usage_count" : 15,
        },
    "111115" :
        {
             "name" : "이원준",
             "age" : 21,
             "last_using_time" : "2024-09-01 00:00:01",
             "usage_count" : 15,
        },
    "111115" :
        {
             "name" : "임수홍",
             "age" : 21,
             "last_using_time" : "2024-09-01 00:00:01",
             "usage_count" : 15,
        },
    "111116" :
        {
             "name" : "임수홍",
             "age" : 21,
             "last_using_time" : "2024-09-01 00:00:01",
             "usage_count" : 15,
        }
}

class FireBase():
    def initializeFirebase(self):
        # Initialize Firebase
        cred = credentials.Certificate("/Users/kevinliam/Desktop/anthony/development/inbody-ai-scale/faceRecognition/src/serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': "https://inbody-scale-ai-default-rtdb.firebaseio.com/",
            'storageBucket': "inbody-scale-ai.appspot.com",
        })

        return db, storage

    def upload_data_to_firebase(self, data):
        db, _ = self.initializeFirebase()
        ref = db.reference('Person')
        
        for key, value in data.items():
            ref.child(key).set(value)

if __name__ == "__main__":
    firebase = FireBase()
    firebase.upload_data_to_firebase(data)
    print("upload image to firebase")
