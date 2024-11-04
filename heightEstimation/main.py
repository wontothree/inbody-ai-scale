import cv2
import os

from gazeTracking.gaze_tracking import GazeTracking
from personSegmentation.personSegmentation import PersonSegmentation

def gaze_tracking_for_image():

    # Initialize GazeTracking
    gaze = GazeTracking()

    # Get the current directory
    current_directory = os.getcwd()

    # List all image files in the current directory
    image_files = [f for f in os.listdir(current_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(current_directory, image_file)
        frame = cv2.imread(image_path)
        
        # If the image was not read successfully, skip it
        if frame is None:
            continue

        gaze.refresh(frame)

        frame = gaze.annotated_frame()


        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()

        cv2.imshow("Demo", frame)
        
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()

def gaze_tracking_for_webcam():

    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        cv2.imshow("Demo", frame)

        if cv2.waitKey(1) == 27:
            break
    
    webcam.release()
    cv2.destroyAllWindows()

def person_segmentation_for_webcam():

    segmenter = PersonSegmentation(model_path='./person_segmentation/deeplabv3.tflite')
    webcam = cv2.VideoCapture(0)

    print("사용 방법:")
    print("'s' 키를 입력하여 세그멘테이션을 수행하고 가장 높은 픽셀을 출력합니다.")
    print("'q' 키를 입력하여 프로그램을 종료합니다.")

    while True:
        # 웹캠에서 새로운 프레임을 가져옴
        ret, frame = webcam.read()

        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # 프레임을 보여줌
        cv2.imshow("Webcam Feed", frame)

        # 's' 키 입력을 기다림
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("Segmentation 시작")
            # 세그멘테이션할 이미지 파일 이름
            image_file_name = 'current_frame.jpg'
            # 현재 프레임을 이미지 파일로 저장
            cv2.imwrite(image_file_name, frame)
            # 세그멘테이션 수행
            category_mask = segmenter.segment_image(image_file_name)
            highest_pixel = segmenter.find_highest_pixel(category_mask)
            print(f'-------------------------------- 가장 높은 픽셀: {highest_pixel} --------------------------------')

        # 'q' 키가 눌리면 종료
        elif key == ord('q'):
            break

    # 웹캠 및 모든 창 닫기
    webcam.release()
    cv2.destroyAllWindows()

def tracking_and_segmentation():
    segmenter = PersonSegmentation(model_path='heightEstimation/personSegmentation/deeplabv3.tflite')
    webcam = cv2.VideoCapture(0)
    gaze = GazeTracking()

    print("사용 방법:")
    print("'s' 키를 입력하여 세그멘테이션을 수행하고 가장 높은 픽셀을 출력합니다.")
    print("'q' 키를 입력하여 프로그램을 종료합니다.")

    while True:

        # We get a new frame from the webcam
        _, frame = webcam.read()

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("Segmentation 시작")
            # 세그멘테이션할 이미지 파일 이름
            image_file_name = 'current_frame.jpg'
            # 현재 프레임을 이미지 파일로 저장
            cv2.imwrite(image_file_name, frame)
            # 세그멘테이션 수행
            category_mask = segmenter.segment_image(image_file_name)
            highest_pixel = segmenter.find_highest_pixel(category_mask)
            print(f'-------------------------------- 가장 높은 픽셀: {highest_pixel} --------------------------------')

        # 'q' 키가 눌리면 종료
        elif key == ord('q'):
            break

        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        cv2.imshow("Demo", frame)

        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # gaze_tracking_for_image()
    # gaze_tracking_for_webcam()
    # person_segmentation_for_webcam()

    tracking_and_segmentation()
