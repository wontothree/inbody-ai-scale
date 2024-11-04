# Deep Face

database

|Id|Name|Sex|Age|EncodingVector|
|---|---|---|---|---|

# Deepface

```py
result  = DeepFace.verify("./111111.png", "./9999991.png")
```

```bash
{'verified': False, 
'distance': 0.9518074983179077, 
'threshold': 0.68, 
'model': 'VGG-Face', 
'detector_backend': 'opencv', 
'similarity_metric': 'cosine', 
'facial_areas': {'img1': {'x': 979, 'y': 2535, 'w': 61, 'h': 61, 'left_eye': None, 'right_eye': None}, 'img2': {'x': 204, 'y': 416, 'w': 840, 'h': 840, 'left_eye': (856, 606), 'right_eye': (477, 759)}}, 
'time': 5.76}
```

```py
analysis = DeepFace.analyze(img_path = "d/111111.png", actions = ["age", "gender", "emotion", "race"])
```

```bash
[{'age': 29, 'region': {'x': 979, 'y': 2535, 'w': 61, 'h': 61, 'left_eye': None, 'right_eye': None}, 'face_confidence': 0.95, 'gender': {'Woman': 14.582684636116028, 'Man': 85.41731238365173}, 'dominant_gender': 'Man', 'emotion': {'angry': 4.4680677354335785, 'disgust': 9.03958380149561e-05, 'fear': 7.966276258230209, 'happy': 0.04098847566638142, 'sad': 53.678345680236816, 'surprise': 0.0011093219654867426, 'neutral': 33.84512662887573}, 'dominant_emotion': 'sad', 'race': {'asian': 12.802787125110626, 'indian': 23.859240114688873, 'black': 13.289110362529755, 'white': 11.504651606082916, 'middle eastern': 22.526249289512634, 'latino hispanic': 16.017967462539673}, 'dominant_race': 'indian'}]
```

```py
DeepFace.stream(db_path="./d")
```