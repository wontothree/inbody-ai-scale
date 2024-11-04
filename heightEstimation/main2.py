import cv2
import os
import numpy as np
from gazeTracking.gaze_tracking import GazeTracking
from personSegmentation.personSegmentation import PersonSegmentation

def process_images_in_directory(directory, eye_positions):
    segmenter = PersonSegmentation(model_path='/Users/kevinliam/Desktop/Kevin’s MacBook Air/development/inbody-scale-ai/heightEstimation/personSegmentation/deeplabv3.tflite')
    result_list = []

    # 디렉토리 내의 모든 이미지 파일을 정렬된 순서로 처리
    filenames = sorted([f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for idx, filename in enumerate(filenames):
        image_path = os.path.join(directory, filename)
        highest_pixel = process_image_segmentation(image_path, segmenter)
        eye_coordinates = process_image(image_path)
        
        if highest_pixel is not None and eye_coordinates[0] is not None and eye_coordinates[1] is not None:
            eye_position_pixel = (eye_coordinates[0][1] + eye_coordinates[1][1]) / 2
            height_estimation = compute_height_estimation(eye_position_pixel, highest_pixel, eye_positions[idx])
            result_list.append(height_estimation)
            print(f'{filename}: Height Estimation = {height_estimation:.2f} cm')
        else:
            # 눈 또는 정수리가 감지되지 않았을 경우 NaN 추가
            result_list.append(np.nan)
            print(f'No person or eyes detected in {filename}')

    return result_list

def process_image_segmentation(image_path, segmenter):
    # 이미지를 읽어오기
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"Error: Could not read image {image_path}.")
        return None

    # 세그멘테이션 수행
    category_mask = segmenter.segment_image(image_path)
    highest_pixel = segmenter.find_highest_pixel(category_mask)

    return highest_pixel

def process_image(image_path):
    gaze = GazeTracking()

    # 이미지 파일을 읽어오기
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}.")
        return None, None

    # 이미지를 GazeTracking에 전달하여 분석
    gaze.refresh(frame)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    return left_pupil, right_pupil

def compute_height_estimation(eye_position_pixel, highest_pixel, eye_position):
    H, W = 640, 480
    DISTANCE = 500  # mm
    VFOV = 55 * np.pi / 180 
    mm_per_pixel = DISTANCE * np.tan(VFOV / 2) / (H / 2)
    cm_per_pixel = mm_per_pixel / 10

    if np.isnan(highest_pixel) or np.isnan(eye_position_pixel):
        pixel_difference = np.nan
    else:
        pixel_difference = -(highest_pixel - eye_position_pixel)
    
    if np.isnan(pixel_difference):
        height_estimation = np.nan
    else:
        height_estimation = eye_position + cm_per_pixel * pixel_difference

    return height_estimation

if __name__ == "__main__":
    # 처리할 이미지 파일 경로
    directory = '/Users/kevinliam/Desktop/Kevin’s MacBook Air/development/inbody-scale-ai/heightEstimation/outputImages'
    eye_positions = [168.2, 171.8, 173.2, 175.2, 177.3, 179.9, 160.5, 163.2, 166.0, 167.4, 169.5, 172.0, 168.1, 169.9, 171.5, 174.0, 176.9, 178.4, 162.5, 164.8, 166.8, 169.1, 171.6, 173.3, 157.0, 159.4, 161.3, 163.4, 165.5, 167.8, 168.9, 170.8, 173.0, 175.0, 176.9, 179.0, 162.4, 165.0, 166.6, 169.1, 171.0, 173.1, 163.4, 165.2, 167.5, 169.7, 172.0, 174.0, 165.0, 167.8, 169.5, 171.9, 174.4, 176.0, 166.6, 168.6, 171.0, 173.0, 174.5, 176.6, 163.4, 165.9, 168.1, 169.9, 172.1, 174.6, 148.9, 151.5, 153.8, 156.0, 157.4, 159.0, 154.0, 156.0, 157.1, 160.5, 161.9, 164.0, 165.7, 167.9, 170.3, 172.1, 174.2, 176.1, 154.3, 156.3, 158.5, 160.5, 162.5, 164.9, 166.6, 169.1, 170.4, 172.1, 174.0, 176.5, 165.6, 167.6, 169.6, 172.3, 174.4, 176.4, 168.3, 170.3, 171.8, 173.1, 176.6, 178.3, 160.0, 162.5, 163.5, 166.7, 169.0, 169.8, 162.0, 163.9, 165.9, 168.3, 169.4, 172.0, 168.4, 170.6, 173.0, 175.3, 177.6, 179.1, 143.1, 145.7, 147.6, 149.9, 151.8, 153.9, 164.0, 166.3, 168.3, 170.0, 172.0, 173.6, 154.0, 157.0, 158.6, 160.5, 163.3, 165.3, 166.9, 169.3, 171.5, 172.9, 174.3, 176.6, 170.1, 172.6, 174.6, 176.5, 179.0, 180.6, 152.1, 154.5, 156.6, 158.6, 160.5, 162.6, 167.6, 169.9, 171.8, 173.5, 176.6, 178.4, 162.0, 164.3, 166.3, 168.0, 171.3, 173.6, 169.1, 171.2, 174.0, 176.4, 178.7, 181.1, 160.6, 162.7, 164.5, 166.4, 169.0, 170.6, 152.5, 154.8, 156.1, 158.6, 160.5, 162.5]

    results = process_images_in_directory(directory, eye_positions)

    print(results)
