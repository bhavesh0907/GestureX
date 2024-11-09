import argparse
import copy
import math
import cv2
import mediapipe as mp
import numpy as np
import vgamepad as vg

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--use_static_image_mode", action='store_true')
    parser.add_argument("--min_detection_confidence",
                      help='min_detection_confidence',
                      type=float,
                      default=0.7)
    parser.add_argument("--min_tracking_confidence",
                      help='min_tracking_confidence',
                      type=float,
                      default=0.5)
    
    return parser.parse_args()

def calc_landmark_list(image, landmarks):
    """Convert landmarks to pixel coordinates."""
    image_width, image_height = image.shape[1], image.shape[0]
    
    landmark_point = []
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point

def draw_landmarks(image, landmark_point):
    """Draw landmarks and connections."""
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Other fingers
        for i in range(9, 21, 4):
            cv2.line(image, tuple(landmark_point[i]), tuple(landmark_point[i+1]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[i]), tuple(landmark_point[i+1]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[i+1]), tuple(landmark_point[i+2]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[i+1]), tuple(landmark_point[i+2]),
                    (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[i+2]), tuple(landmark_point[i+3]),
                    (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[i+2]), tuple(landmark_point[i+3]),
                    (255, 255, 255), 2)

    # Draw points
    for point in landmark_point:
        cv2.circle(image, tuple(point), 8, (255, 255, 255), -1)
        cv2.circle(image, tuple(point), 8, (0, 0, 0), 1)

    return image

def draw_joystick_zone(image, origin, radius, color):
    """Draw joystick origin and control zone with specified color."""
    cv2.circle(image, tuple(origin), radius, color, 2)
    cv2.circle(image, tuple(origin), 5, (0, 255, 0), -1)
    return image

class HandController:
    def __init__(self, is_right_hand=True):
        self.is_right_hand = is_right_hand
        self.origin = [800, 200] if is_right_hand else [400, 200]
        self.deadzone = 40
        self.sensitivity = 1
        self.radius = 150
        self.color = (0, 0, 255) if is_right_hand else (255, 0, 0)  # Red for right, Blue for left
        
    def process_joystick(self, landmark_list, gamepad):
        """Process joystick movement based on hand landmarks."""
        if not landmark_list:
            return
            
        # For right hand - use index finger (point 8) for camera control
        # For left hand - use thumb tip (point 4) for movement control
        control_point = 8 if self.is_right_hand else 4
        
        if math.dist(self.origin, landmark_list[control_point]) > self.deadzone:
            diff = list(map(lambda x, y: x-y, self.origin, landmark_list[control_point]))
            coords = list(map(lambda x: (x / self.radius) * self.sensitivity, diff))
            coords = np.clip(coords, -1.0, 1.0)
            coords[0] *= -1
            
            if self.is_right_hand:
                gamepad.right_joystick_float(x_value_float=coords[0], y_value_float=coords[1])
            else:
                gamepad.left_joystick_float(x_value_float=coords[0], y_value_float=coords[1])
        else:
            if self.is_right_hand:
                gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
            else:
                gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)

def main():
    args = get_args()
    
    # Initialize controllers for each hand
    right_controller = HandController(is_right_hand=True)
    left_controller = HandController(is_right_hand=False)
    
    gamepad = vg.VX360Gamepad()
    pressed_tolerance = 20

    # Initialize camera
    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    while True:
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break

        # Settings adjustment
        if key == ord('w'):
            try:
                print("\nRight Hand Settings:")
                right_controller.deadzone = int(input("Enter right deadzone size: "))
                right_controller.sensitivity = float(input("Enter right sensitivity: "))
                right_controller.radius = int(input("Enter right radius: "))
                
                print("\nLeft Hand Settings:")
                left_controller.deadzone = int(input("Enter left deadzone size: "))
                left_controller.sensitivity = float(input("Enter left sensitivity: "))
                left_controller.radius = int(input("Enter left radius: "))
            except ValueError:
                print("Invalid input, keeping current values")

        ret, image = cap.read()
        if not ret:
            break
            
        image = cv2.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Process hands
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                handedness = handedness.classification[0].label[0:]
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Reset joystick origin
                if key == ord('q'):
                    if handedness == "Right":
                        right_controller.origin = landmark_list[8]
                        print(f"Right Joystick Origin: {right_controller.origin}")
                    if handedness == "Left":
                        left_controller.origin = landmark_list[4]  # Use thumb for left hand
                        print(f"Left Joystick Origin: {left_controller.origin}")

                controller = right_controller if handedness == "Right" else left_controller
                controller.process_joystick(landmark_list, gamepad)

                # Button controls for right hand
                if handedness == "Right":
                    # ABXY buttons using finger-thumb pinch
                    if math.dist(landmark_list[8], landmark_list[4]) < pressed_tolerance:
                        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
                    else:
                        gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)

                    if math.dist(landmark_list[12], landmark_list[4]) < pressed_tolerance:
                        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
                    else:
                        gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)

                # Trigger controls for left hand
                if handedness == "Left":
                    # Triggers using finger-thumb pinch
                    if math.dist(landmark_list[8], landmark_list[4]) < pressed_tolerance:
                        gamepad.right_trigger(value=255)
                    else:
                        gamepad.right_trigger(value=0)

                    if math.dist(landmark_list[12], landmark_list[4]) < pressed_tolerance:
                        gamepad.left_trigger(value=255)
                    else:
                        gamepad.left_trigger(value=0)

                # Draw visualization
                debug_image = draw_landmarks(debug_image, landmark_list)

        # Always draw joystick zones
        debug_image = draw_joystick_zone(debug_image, right_controller.origin, 
                                       right_controller.radius, right_controller.color)
        debug_image = draw_joystick_zone(debug_image, left_controller.origin, 
                                       left_controller.radius, left_controller.color)

        gamepad.update()
        cv2.imshow('Hand Gesture Controller', debug_image)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()