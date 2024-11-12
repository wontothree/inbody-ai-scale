# Face Identification Pipline

    deepface
    ├── registration_imgs/                          # images folder for registration
    ├── search_imgs/                                # images folder for search
    ├── src/
    |     ├── capture.py                            # image capture
    |     ├── faceRecognition.py                    # face recognition using deepface library
    |     ├── firebase.py                           # database
    |     ├── registration.py                       # user registration
    |     ├── search.py                             # user search

# Pipeline

1. [Registration] 사용자 정보 [id, name, sex, age, encoding]를 입력받고, database1에 올린다.
2. [InBody AI Scale] 사용자 얼굴을 촬영하고 database2에 올린다.
3. [Searching] database2에 올라온 image를 조회한다.
4. [Searching] image에 대한 embedding vector를 구하고 database1에서 조회하여 해당하는 user information을 찾는다.
5. [Searching] user inoformation을 database3에 올린다.
6. [InBody AI Scale] 최근에 올라온 user information을 조회한다.

# Firebase

database1

|Id|Name|Sex|Age|EmbeddingVector|
|---|---|---|---|---|
