import cv2
import time

from firebase import FireBase

class Capture:
    def capture_photo(self, filename='search_imgs/tmp.png'):
        """
        capture_photo(self, filename='search_imgs/tmp.png')

        This function captures a photo using the camera and provides an interface for the user to decide whether to save it. 
        By default, the captured image is saved to the specified file path.

        Parameters:
        - filename (str): The file path and name where the photo will be saved (default: 'search_imgs/tmp.png').

        Functionality:
        1. Initializes the camera and checks if it is accessible.
        2. Guides the user to press 's' to save the photo or 'q' to exit without saving.
        3. Continuously reads frames from the camera and displays them on the screen.
        4. If the user presses 's', the photo is saved to the specified path, and the process exits.
        5. If the user presses 'q', the process exits without saving.

        Note:
        - The file save operation may fail if the specified directory does not exist.
        """

        # 카메라 초기화
        cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 의미합니다.

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
        
        print("complete capture")


if __name__ == "__main__":
    capture = Capture()
    firebase = FireBase()

    # capture face image
    capture.capture_photo()

    time.sleep(3)

    firebase.upload_image_to_firebase_storage('search_imgs/tmp.png')