from deepface import DeepFace
import numpy as np

class FaceRecognition:
    def __init__(self):
        pass

    def generate_embedding(self, img_path):
        object = DeepFace.represent(
            img_path = img_path, model_name="Facenet", detector_backend="opencv"
        )

        embedding = object[0]["embedding"]

        print("complete generating face embedding")

        return embedding
    
    def generate_cosine_similarity(self, embedding1, embedding2):

        dot_product = np.dot(embedding1, embedding2)
        embedding1_magnitude = np.linalg.norm(embedding1)
        embedding2_magnitude = np.linalg.norm(embedding2)

        cosine_similarity = dot_product / (embedding1_magnitude * embedding2_magnitude)

        return cosine_similarity


if __name__ == "__main__":
    face_recognition = FaceRecognition()

    print(face_recognition.generate_embedding("./deepface/imgs/111111.png"))

