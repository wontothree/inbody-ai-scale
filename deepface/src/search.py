from capture import Capture
from faceRecognition import FaceRecognition
from firebase import FireBase

class Search:
    def __init__(self):
        self.face_recognition = FaceRecognition()

    def search_user(self, embedding, dataset):
        cosine_similarity_list = []
        names = []
        for data in dataset:
            if data is  None:
                continue

            data_embedding = data["embedding"]
            cosine_similarity = self.face_recognition.generate_cosine_similarity(embedding, data_embedding)
            cosine_similarity_list.append(cosine_similarity)
            names.append(data["name"])
        
        similarity_results = list(zip(names, cosine_similarity_list))

        for name, similarity in similarity_results:
            print(f"Name: {name}, Cosine Similarity: {similarity}")

        max_index = cosine_similarity_list.index(max(cosine_similarity_list))

        return names[max_index]



if __name__ == "__main__":
    search = Search()
    capture = Capture()
    face_recognition = FaceRecognition()
    firebase = FireBase()

    # capture user image
    capture.capture_photo('./deepface/search_imgs/tmp.png')

    # extract embedding
    embedding = face_recognition.generate_embedding('./deepface/search_imgs/tmp.png')

    # download all user data from firebase
    dataset = firebase.get_all_data_from_firebase()

    # search
    user_name = search.search_user(embedding, dataset)

    print("result", user_name)
