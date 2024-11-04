# Research Note

다음의 코드에서 두 출력은 완벽히 동일하다.

```py
from deepface import DeepFace
import numpy as np

img_path1 = "test1/111111.png"
img_path2 = "test1/111112.png"

# target embedding
obj1 = DeepFace.represent(
    img_path = img_path1, model_name="Facenet", detector_backend="opencv"
)

obj2 = DeepFace.represent(
    img_path = img_path2, model_name="Facenet", detector_backend="opencv"
)

target_embedding1 = obj1[0]["embedding"]
target_embedding2 = obj2[0]["embedding"]


dot_product = np.dot(target_embedding1, target_embedding2)
magnitude_A = np.linalg.norm(target_embedding1)
magnitude_B = np.linalg.norm(target_embedding2)

cosine_similarity = dot_product / (magnitude_A * magnitude_B)

print(f"Cosine Similarity using NumPy: {cosine_similarity}")

result = DeepFace.verify(
    img1_path=img_path1,
    img2_path=img_path2,
    model_name="Facenet",
    detector_backend="opencv",
    distance_metric="cosine"
)

# 코사인 거리와 코사인 유사도 출력
cosine_distance = result["distance"]
cosine_similarity = 1 - cosine_distance  # 코사인 유사도로 변환

print(f"Cosine Similarity: {cosine_similarity}")
```