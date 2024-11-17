import cv2
import numpy as np
import mediapipe as mp

class HandGestureDetector:
    def __init__(self):
        # 初始化MediaPipe手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def detect_gesture(self, frame):
        # 转换颜色空间从BGR到RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 处理图像
        results = self.hands.process(frame_rgb)
        
        # 初始化手势状态
        gesture = "Unknown"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 在图像上绘制手部关键点
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # 获取所有关键点的坐标
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y])
                
                # 检测简单手势
                gesture = self.recognize_gesture(landmarks)
                
                # 在图像上显示手势类型
                cv2.putText(
                    frame,
                    f"Gesture: {gesture}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
        return frame, gesture
    
    def recognize_gesture(self, landmarks):
        # 获取手指尖点(食指和中指)
        index_finger_tip = landmarks[8]
        middle_finger_tip = landmarks[12]
        
        # 获取手掌基准点
        wrist = landmarks[0]
        
        # 计算手指尖到手腕的距离
        index_distance = self.calculate_distance(index_finger_tip, wrist)
        middle_distance = self.calculate_distance(middle_finger_tip, wrist)
        
        # 简单的手势判断逻辑
        if index_distance > 0.3 and middle_distance > 0.3:
            return "Peace"
        elif index_distance > 0.3 and middle_distance <= 0.3:
            return "One"
        else:
            return "Fist"
    
    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    detector = HandGestureDetector()
    
    while True:
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            break
            
        # 水平翻转图像(镜像效果)
        frame = cv2.flip(frame, 1)
        
        # 检测手势
        frame, gesture = detector.detect_gesture(frame)
        
        # 显示结果
        cv2.imshow('Hand Gesture Detection', frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()