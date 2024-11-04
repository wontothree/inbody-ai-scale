import cv2

class FaceRegistration:

    def get_user_info(self):
        # Name
        name = input("Name: ")
        # Sex
        sex = input("Sex: ")
        # Age
        age = input("Age: ")
        
        return name, sex, age

    def is_user_registered(self, name, sex, age, encoding_vector):
        # Simulated registration list
        registration_list = [
            {"name": "Sewon", "sex": "Man", "age": "23", "encoding_vector" : 1},
            {"name": "Elon", "sex": "Man", "age": "53", "encoding_vector" : 2}
        ]

        # Check if the provided information matches any in the registration list
        for user in registration_list:
            if (user["name"] == name and
                user["sex"] == sex and
                user["age"] == age and
                user["encoding_vector"] == encoding_vector):
                return True
        return False
    
    def capture_photo(self, filename='captured_photo.jpg'):
        # 카메라 초기화
        cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다. 다른 카메라를 사용할 경우 인덱스를 변경할 수 있습니다.

        if not cap.isOpened():
            print("Error: Camera not accessible.")
            return

        print("Press 's' to save the photo or 'q' to quit without saving.")

        while True:
            # 카메라로부터 프레임을 읽습니다.
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image.")
                break

            # 프레임을 화면에 표시합니다.
            cv2.imshow('Capture Photo', frame)

            # 사용자 입력을 처리합니다.
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # 's' 키가 눌리면 사진을 저장합니다.
                cv2.imwrite(filename, frame)
                print(f"Photo saved as {filename}.")
                break
            elif key == ord('q'):
                # 'q' 키가 눌리면 프로그램을 종료합니다.
                print("Exiting without saving.")
                break

        # 자원 해제
        cap.release()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    face_registration = FaceRegistration()

    # 1
    name, sex, age = face_registration.get_user_info()
    print(f"Name: {name}")
    print(f"Sex: {sex}")
    print(f"Age: {age}")

    # 2
    if face_registration.is_user_registered(name, sex, age, 1):
        print("Already Registered")
    else:
        print("Not Registered")

    # 3
    face_registration.capture_photo('captured_photo.jpg')
