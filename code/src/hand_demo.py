import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

# 配置手部检测选项
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# 手部关键点连线关系
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

POINT_COLOR = (0, 255, 0)
LINE_COLOR = (255, 0, 0)

def calculate_angle(p1, p2, p3):
    """计算三点之间的角度（p2为顶点），用于判断手指是否伸直"""
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    norm_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    norm_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle = math.degrees(math.acos(cos_angle))
    return angle

def count_fingers_robust(hand_landmarks):
    """基于角度判断手指伸出数量"""
    count = 0
    # 拇指：指尖(4)、指关节(3)、手腕(0)
    if calculate_angle(hand_landmarks[4], hand_landmarks[3], hand_landmarks[0]) > 150:
        count += 1
    # 其余四指：指尖、第二关节、手腕(0)
    for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if calculate_angle(hand_landmarks[tip_idx], hand_landmarks[pip_idx], hand_landmarks[0]) > 150:
            count += 1
    return count

def recognize_rps(hand_landmarks):
    """识别石头、剪刀、布，识别不出则返回空字符串"""
    finger_count = count_fingers_robust(hand_landmarks)
    if finger_count == 0:
        return "ROCK"
    elif finger_count == 2:
        return "SCISSORS"
    elif finger_count == 5:
        return "PAPER"
    else:
        return ""  # 修改点1：不满足条件时返回空字符串

print("正在启动摄像头，按 'q' 键退出...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("摄像头读取失败")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            # 获取左右手信息并修正镜像反转
            handedness = detection_result.handedness[idx]
            hand_label = handedness[0].category_name
            if hand_label == "Left":
                hand_label = "RIGHT HAND"
            else:
                hand_label = "LEFT HAND"

            # 进行手指计数和石头剪刀布识别
            rps_gesture = recognize_rps(hand_landmarks)
            finger_count = count_fingers_robust(hand_landmarks)

            # 绘制手部骨架关键点
            for landmark in hand_landmarks:
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 4, POINT_COLOR, -1)

            # 绘制手部骨架连线
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                sp = hand_landmarks[start_idx]
                ep = hand_landmarks[end_idx]
                cv2.line(frame,
                         (int(sp.x * w), int(sp.y * h)),
                         (int(ep.x * w), int(ep.y * h)),
                         LINE_COLOR, 2)

            # 获取手腕位置，用于放置文字标签
            wrist = hand_landmarks[0]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)

            # --- 文字显示逻辑 ---
            # 1. 显示左右手标签 (始终显示)
            cv2.putText(frame, hand_label,
                        (wrist_x - 50, wrist_y - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 2. 显示石头剪刀布结果 (只有识别出结果时才显示)
            if rps_gesture != "":  # 修改点2：增加判断，非空才绘制
                cv2.putText(frame, rps_gesture,
                            (wrist_x - 40, wrist_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # 3. 显示手指数量 (始终显示)
            cv2.putText(frame, f"Fingers: {finger_count}",
                        (wrist_x - 50, wrist_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, "Hand Detection - RPS Demo", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Hand Detection - RPS Demo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("程序已退出")