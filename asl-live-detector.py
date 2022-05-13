import cv2
import mediapipe as mp
import time
import numpy as np
import keras
import pickle
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


class handDetector():
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def get_world_keypoints(self):
        if not self.results.multi_hand_landmarks:
            return None

        keypoints = []
        for j in range(21):
            landmark = self.results.multi_hand_world_landmarks[0].landmark[j]
            keypoints.append([landmark.x, landmark.y, landmark.z])
            
        return np.array(keypoints).reshape(-1)


    def findPosition(self, img, handNo = 0, draw = True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

        lmarr = np.array(lmlist)

        x_min, y_min, x_max, y_max = -1, -1, -1, -1

        if len(lmarr.shape) >= 2:
            x_min, x_max = np.min(lmarr[:,1]), np.max(lmarr[:,1])
            y_min, y_max = np.min(lmarr[:,2]), np.max(lmarr[:,2])

            x_min -= 25
            y_min -= 25
            x_max += 25
            y_max += 25

            height = y_max - y_min
            width = x_max - x_min
            add_factor = (height-width)/2

            x_min -= int(np.floor(add_factor))
            x_max += int(np.ceil(add_factor))

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        return lmlist, (x_min, y_min, x_max, y_max)


def main():    
    MODEL_PATH = 'mnist_cnn.keras'
    model = keras.models.load_model(MODEL_PATH)
    print(model.summary())

    with open('keypoints_svm.pkl','rb') as f:
        clf = pickle.load(f)

    print(clf)
    
    dsize = (28,28)

    pTime = 0
    cTime = 0
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = handDetector()

    while True:
        success, img = cam.read()

        frame = np.copy(img)
        
        # HAND DETECTOR STUFF
        img = detector.findHands(img, draw=True)
        lmlist, (x_min, y_min, x_max, y_max) = detector.findPosition(img, draw=True)
        if len(lmlist) != 0:
            print(lmlist[4])

        keypoints = detector.get_world_keypoints()
        if keypoints is not None:
            y_pred = clf.predict(np.expand_dims(keypoints, axis=0))[0]
            print(y_pred)
            letter = chr(y_pred+65) 
            print(letter)

            cv2.putText(img, letter, (10, 300), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        

        # MODEL STUFF
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (x_min != -1):
            frame = frame[y_min:y_max, x_min:x_max]

        if len(lmlist) != 0 and frame.shape[0] != 0 and frame.shape[1] != 0:

            # cv2.imshow("Frame", frame)

            frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_NEAREST)
            
            frame = frame / 255 # normalize

            # convert to (1,N,N,1)
            frame = np.expand_dims(frame, axis=2)
            frame = np.expand_dims(frame, axis=0)
            # prediction ready frame now
                
            probs = model.predict(frame)[0]
            y_pred = np.argmax(probs)
            letter = chr(y_pred+65)    


            cv2.putText(img, letter, (10, 200), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
            
            # img = cv2.flip(img, 1)

        cv2.imshow("Image", img)

        k = cv2.waitKey(1)

        if k % 256 == 27: # close on escape key
            break


if __name__ == "__main__":
    main()