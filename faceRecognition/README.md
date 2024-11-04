# Face Recognition

    faceRecognition
    ├── imgs/                                       # images folder
    ├── src/            
    |     ├── faceIdentification.py                 # opencv and face_recognition codes
    |     ├── firebase.py                           # add data to Firebase
    |     ├── encodeGenerator.py                    # firebase storage에서 저장된 사진으로 encoding을 추출하여 .p 파일로 저장한다.
    |     ├── serviceAccountKey.json                # authority for Firebase
    |     ├── Encode.p                              # 

# Database

Firebase

- Realtime Database : meta data
- Storage : image data

# Pipeline

## Face Registration

등록하는 과정에서 이미 등록된 사용자는 등록되어 있음을 알린다.

1. Name, sex, age를 입력한다.
2. 이미 등록된 사용자에 대해 이미 등록되어 있음을 알린다.
3. 사진을 찍는다. 이때 사용자가 제대로 된 사진을 찍을 수 있도록 유도한다.
4. 사진에서 얼굴의 encoding vector를 추출한다.
5. 각 사람에 대해 id를 부여한다.
6. Id, name, sex, age, image, encoding_vector, usage_count, last_usage_time을 database에 저장한다.
7. 등록 완료 메시지를 전달한다.

database

|Id|Name|Sex|Age|EncodingVector|
|---|---|---|---|---|

## Face Identification

1. 사진을 촬영한다.
2. 얼굴 인식 및 encoding
3. database에 있는 encodeing과 cosine similarity를 비교한다.
4. 해당하는 encoding을 찾고 사용자 정보를 반환한다.

# Dependencies

```bash
pip install -r requirements.txt

pip install numpy

pip install opencv-python

pip install face_recognition

pip install pickle

pip install firebase_admin
```

# Research Note

2024.09.24

문제 Jetson Nano에서 모델 추론을 하는 데 너무 많은 시간이 소요되며 부정확하다. 카메라를 켜는 순간부터 버벅거림이 심하다.

Jetson Nano는 RAM이 4GB밖에 없어서 dlib을 컴파일하는 데 충분하지 않습니다. 이를 해결하기 위해 swapfile을 생성하여 디스크 공간을 추가 RAM처럼 사용할 수 있게 설정할 것입니다.

```bash
git clone https://github.com/JetsonHacksNano/installSwapfile

./installSwapfile/installSwapfile.sh
```
