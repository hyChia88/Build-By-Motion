import cv2
import numpy as np
import mediapipe as mp

class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        
        self.prev_x = None
        self.counter = 0
        self.swipe_threshold = 0.05

    def detect_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # 获取图像尺寸用于坐标转换
        frame_height, frame_width = frame.shape[:2]
        
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            
            # 方法1：显示特定点（食指尖端-索引8）
            # 将手部关键点的相对坐标转换为图像上的像素坐标
            index_finger_x = int(hand.landmark[8].x * frame_width)
            index_finger_y = int(hand.landmark[8].y * frame_height)
            
            # 在食指尖端画一个红色圆点
            cv2.circle(frame, 
                      (index_finger_x, index_finger_y), # 坐标
                      10,  # 半径
                      (0, 0, 255),  # 颜色(BGR) - 红色
                      -1)  # -1表示填充圆
            
            # 方法2：显示所有手部关键点
            for id, landmark in enumerate(hand.landmark):
                # 将相对坐标转换为像素坐标
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                # 画小圆点
                cv2.circle(frame, 
                          (x, y),
                          5,  # 半径
                          (255, 0, 0),  # 蓝色
                          -1)  # 填充
                
                # 显示关键点的索引号
                cv2.putText(frame, 
                           str(id),  # 显示索引号
                           (x+10, y+10),  # 位置稍微偏移以免遮挡点
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,  # 字体大小
                           (255, 255, 255),  # 白色
                           1)  # 线条粗细
            
            # 手势检测逻辑
            current_x = hand.landmark[8].x
            
            if self.prev_x is not None:
                movement = current_x - self.prev_x
                if abs(movement) > self.swipe_threshold:
                    if movement > 0:
                        self.counter -= 1
                        gesture = "Left to Right"
                    else:
                        self.counter += 1
                        gesture = "Right to Left"
                else:
                    gesture = "Hand detected"
            else:
                gesture = "Hand detected"
            
            self.prev_x = current_x
            self._draw_info(frame, gesture)
            
        else:
            self.prev_x = None
            self._draw_info(frame, "No hand detected")
        
        return frame
    
    def _draw_info(self, frame, gesture):
        cv2.putText(frame, f"Count: {self.counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(0)
    detector = HandGestureDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = detector.detect_gesture(frame)
        cv2.imshow('Hand Gesture Counter', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()