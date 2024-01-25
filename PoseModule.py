import cv2
import mediapipe as mp
import tensorflow as tf
import time

class poseDetector():

    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode 
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw = True):
        # Pose detection
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img



    
        # for id, lm in enumerate(results.pose_landmarks.landmark):
        #     h, w, c = img.shape
        #     print(id, lm)
        #     cx, cy = int(lm.x * w), int(lm.y * h)
        #     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # Calculate and display FPS for pose detection
    # cTimePose = time.time()
    # fpsPose = 1 / (cTimePose - pTimePose)
    # pTimePose = cTimePose

    # cv2.putText(img, f'Pose FPS: {int(fpsPose)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    # Calculate and display FPS for video display


def main():
    cap = cv2.VideoCapture('dada/AI_trainer/curls3.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)

        
        cTimeDisplay = time.time()
        fpsDisplay = 1 / (cTimeDisplay - pTime)
        pTime = cTimeDisplay

        cv2.putText(img, f'Display FPS: {int(fpsDisplay)}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)  # Add a slight delay

        frame_count += 1

        # Add break condition to exit the loop after processing max_frames
        if frame_count >= max_frames:
            break

if __name__ == "__main__":
    main()

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
