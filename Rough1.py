import math
import time
import cv2 as cv
import mediapipe as mp


def detect_waist_knee_ankle(image, pose):
    h, w, c = image.shape
    landmarks = pose.process(image).pose_landmarks
    if landmarks:
        left_waist_x = int(landmarks.landmark[24].x * w)
        left_waist_y = int(landmarks.landmark[24].y * h)
        left_knee_x = int(landmarks.landmark[26].x * w)
        left_knee_y = int(landmarks.landmark[26].y * h)
        left_ankle_x = int(landmarks.landmark[28].x * w)
        left_ankle_y = int(landmarks.landmark[28].y * h)
        right_waist_x = int(landmarks.landmark[23].x * w)
        right_waist_y = int(landmarks.landmark[23].y * h)
        right_knee_x = int(landmarks.landmark[25].x * w)
        right_knee_y = int(landmarks.landmark[25].y * h)
        right_ankle_x = int(landmarks.landmark[27].x * w)
        right_ankle_y = int(landmarks.landmark[27].y * h)
        if landmarks.landmark[26].z < landmarks.landmark[25].z:
            return left_waist_x, left_waist_y, left_knee_x, left_knee_y, left_ankle_x, left_ankle_y
        return right_waist_x, right_waist_y, right_knee_x, right_knee_y, right_ankle_x, right_ankle_y
    return None


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))


def calculate_angle_in_degrees(w_x, w_y, k_x, k_y, a_x, a_y):
    d_wk = calculate_distance(w_x, w_y, k_x, k_y)
    d_ka = calculate_distance(k_x, k_y, a_x, a_y)
    d_aw = calculate_distance(a_x, a_y, w_x, w_y)
    angle = math.acos((pow(d_wk, 2) + pow(d_ka, 2) - pow(d_aw, 2)) / (2.0 * d_wk * d_ka))
    angle = round(angle * 180 / math.pi, 3)
    return angle


def display_points_connections_and_angle(image, x1, y1, x2, y2, x3, y3, angle):
    image = cv.circle(image, (x1, y1), 3, (255, 0, 0), -1)
    image = cv.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    image = cv.circle(image, (x2, y2), 3, (0, 255, 0), -1)
    image = cv.line(image, (x2, y2), (x3, y3), (255, 255, 255), 1)
    image = cv.circle(image, (x3, y3), 3, (0, 0, 255), -1)
    image = cv.putText(image, f'{angle}', (x2, y2), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    return image


cap = cv.VideoCapture(r'KneeBendVideo.mp4')
frame_width = 500
frame_height = 500
temp = cv.VideoWriter('Results.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))
p = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.8, min_tracking_confidence=0.8)
ang = 180
start = time.time()
end = time.time()
counter = 0

while True:
    success, frame = cap.read()
    if success:
        frame = cv.resize(frame, (frame_width, frame_height))
        frame = cv.putText(frame, f'REPS COMPLETED: {counter}', (50, 480), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        leg = detect_waist_knee_ankle(frame, p)
        if leg:
            waist_x, waist_y, knee_x, knee_y, ankle_x, ankle_y = leg
            ang = calculate_angle_in_degrees(waist_x, waist_y, knee_x, knee_y, ankle_x, ankle_y)
            if ang < 140:
                frame = cv.putText(frame, 'KEEP YOUR LEG BENT FOR 8 SECONDS', (100, 30), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                start = time.time()
                rep_time = round(start - end, 3)
                frame = cv.putText(frame, f'TIMER: {rep_time}', (300, 480), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                if rep_time >= 8:
                    counter = counter + 1
                    start = time.time()
                    end = time.time()
            else:
                end = time.time()
                start = end
            frame = display_points_connections_and_angle(frame, waist_x, waist_y, knee_x, knee_y, ankle_x, ankle_y, ang)
        cv.imshow('Frame', frame)
        temp.write(frame)
        if cv.waitKey(1) == 27:
            break
    else:
        cap.release()
        temp.release()
        break
