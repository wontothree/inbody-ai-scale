from datetime import datetime

from capture import Capture
from faceRecognition import FaceRecognition
from firebase import FireBase

# threshold for cosine similarity
LOWER_BOUND_THRESHOLD = 0.5
UPPER_BOUND_THRESHOLD = 0.7

class Search:
    def __init__(self):
        self.face_recognition = FaceRecognition()

    def search_max_cosine_similarity_user(self, embedding, dataset):
        cosine_similarity_list = []
        names = []
        uuids = []  # uuid를 저장할 리스트 추가
        
        for uuid, data in dataset.items():  # dataset에서 uuid와 데이터를 동시에 다루기
            if data is None:
                continue

            data_embedding = data["embedding"]
            cosine_similarity = self.face_recognition.generate_cosine_similarity(embedding, data_embedding)
            cosine_similarity_list.append(cosine_similarity)
            names.append(data["name"])
            uuids.append(uuid)  # uuid 추가

        similarity_results = list(zip(names, cosine_similarity_list, uuids))  # uuid도 포함하여 결과 생성

        # print
        for name, similarity, uuid in similarity_results:
            print(f"- Name: {name}, Cosine Similarity: {similarity}, UUID: {uuid}")

        max_index = cosine_similarity_list.index(max(cosine_similarity_list))

        # upper bound threshold
        if (max(cosine_similarity_list) > UPPER_BOUND_THRESHOLD):
            return names[max_index], uuids[max_index]  # 이름과 UUID 반환
        else:
            return None


    def search_recent_threshold_user(self, embedding, dataset):

        # list
        sorted_data = sorted(
            dataset.items(),  # items() 사용하여 uuid와 데이터 모두 접근
            key=lambda x: datetime.strptime(x[1]["time"], "%Y-%m-%d %H:%M:%S"),
            reverse=True
        )

        for user_uuid, data in sorted_data:
            data_embedding = data["embedding"]
            cosine_similarity = self.face_recognition.generate_cosine_similarity(embedding, data_embedding)

            if (cosine_similarity >= LOWER_BOUND_THRESHOLD):
                return user_uuid, data["name"]
        
        return self.search_max_cosine_similarity_user(embedding, dataset)

if __name__ == "__main__":
    search = Search()
    capture = Capture()
    face_recognition = FaceRecognition()
    firebase = FireBase()

    # capture user image
    # capture.capture_photo('./deepface/search_imgs/tmp.png')

    # download user image
    # firebase.download_image_from_firebase_storage(
    #     'search_imgs/tmp.png',
    #     'downloaded_imgs/downloaded_tmp.png'
    # )

    # path of target image
    img_path = '/Users/kevinliam/Desktop/anthony/development/inbody-ai-scale/deepface/registration_imgs/091a85b2-8186-43c3-bf6f-8ec1404f7c94.png'
    
    # extract embedding
    embedding = face_recognition.generate_embedding(img_path)

    # download all user data from firebase
    dataset = firebase.get_all_data_from_firebase()

    # for just print log (not search)
    search.search_max_cosine_similarity_user(embedding, dataset)

    # search
    user_uuid, name = search.search_recent_threshold_user(embedding, dataset)

    print("result: ", user_uuid, ", ", name)
