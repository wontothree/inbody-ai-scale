from datetime import datetime

from capture import Capture
from faceRecognition import FaceRecognition
from firebase import FireBase

# 1. 최근 순으로 탐색하는데 RECENT_THRESHOLD 이상이라면 바로 반환한다.
# 2. 전부 RECENT_THRESHOLD 이하라면, 가장 cosine similarity가 높은 대상을 반환한다. 단 그것도 LOWER_BOUND_THRESHOLD 이상이어야 한다. 등록을 안 한 사람이 측정하는 경우를 고려하기 위함이다.

# if RECENT_THRESHOLD == 0.0, lastly registered person is returned
# if RECENT_THRESHOLD == 1.0, non-bias
RECENT_THRESHOLD = 1.0 # 0.75

# threshold for cosine similarity
# if LOWER_BOUND_THRESHOLD == 0.0, non-bias
# if LOWER_BOUND_THRESHOLD == 1.0,
LOWER_BOUND_THRESHOLD = 0.0 # 0.5

class Search:
    def __init__(self):
        self.face_recognition = FaceRecognition()

    def search_max_cosine_similarity_user(self, embedding, dataset):
        """
        search_max_cosine_similarity_user(self, embedding, dataset)

        This function searches for the user with the highest cosine similarity to a given embedding within a dataset. 
        It calculates the cosine similarity between the provided embedding and each user's embedding in the dataset, 
        and returns the name and UUID of the user with the highest similarity, 
        if the similarity exceeds a specified threshold.

        Parameters:
        - embedding (list or array): The embedding of the face to compare against the dataset.
        - dataset (dict): A dictionary containing user data, where each item contains a user's UUID, name, and embedding.

        Returns:
        - tuple: A tuple containing the name and UUID of the user with the highest cosine similarity.
        If no user exceeds the threshold, returns None.

        Process:
        1. Iterate through the dataset and compute the cosine similarity between the provided embedding and each user's embedding.
        2. Store the similarities along with user names and UUIDs.
        3. Sort and find the user with the highest cosine similarity.
        4. If the highest similarity exceeds a defined threshold, return the user's name and UUID; otherwise, return None.

        Note:
        - The function uses the LOWER_BOUND_THRESHOLD to filter out users with low cosine similarity.

        """

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

        # lower bound threshold
        if (max(cosine_similarity_list) >= LOWER_BOUND_THRESHOLD):
            return names[max_index], uuids[max_index]  # 이름과 UUID 반환
        else:
            return None, None


    def search_recent_threshold_user(self, embedding, dataset):
        """
        search_recent_threshold_user(self, embedding, dataset)

        This function searches for a user in the dataset whose embedding has the highest cosine similarity 
        to the given embedding and whose similarity exceeds a specified upper bound threshold. 
        The dataset is sorted by the 'time' field, and it checks users in reverse chronological order 
        (i.e., most recent first). If no user exceeds the threshold, it falls back to searching 
        for the user with the maximum cosine similarity.

        Parameters:
        - embedding (list or array): The embedding of the face to compare against the dataset.
        - dataset (dict): A dictionary containing user data, where each item contains a user's UUID, name, embedding, and time.

        Returns:
        - tuple: A tuple containing the UUID and name of the user with the highest cosine similarity 
        that exceeds the upper bound threshold, or it calls another function to find the user with the maximum similarity.

        Process:
        1. Sort the dataset by the 'time' field in descending order (most recent data first).
        2. For each user, calculate the cosine similarity between the provided embedding and the user's embedding.
        3. If a user's similarity is greater than or equal to the upper bound threshold, return the user's UUID and name.
        4. If no user exceeds the threshold, the function calls `search_max_cosine_similarity_user` to find the user with the maximum similarity.

        Note:
        - The `RECENT_THRESHOLD` is used to filter out users with low cosine similarity.
        """

        # list
        sorted_data = sorted(
            dataset.items(),  # items() 사용하여 uuid와 데이터 모두 접근
            key=lambda x: datetime.strptime(x[1]["time"], "%Y-%m-%d %H:%M:%S"),
            reverse=True
        )

        for user_uuid, data in sorted_data:
            data_embedding = data["embedding"]
            cosine_similarity = self.face_recognition.generate_cosine_similarity(embedding, data_embedding)

            if (cosine_similarity > RECENT_THRESHOLD):
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
    img_path = '/Users/kevinliam/Desktop/anthony/development/inbody-ai-scale/deepface/search_imgs/jickwon.jpeg'
    
    # extract embedding
    embedding = face_recognition.generate_embedding(img_path)

    # download all user data from firebase
    dataset = firebase.get_all_data_from_firebase()

    # search max cosine similarity
    # for log
    # search.search_max_cosine_similarity_user(embedding, dataset)

    # search
    user_uuid, name = search.search_recent_threshold_user(embedding, dataset)

    print("result: ", user_uuid, ", ", name)
