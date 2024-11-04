import cv2

class Capture:
    def __init__(self):
        pass

    def capture_photo(self, filename='./registration_imgs/tmp.png'):
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
