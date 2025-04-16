import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time 
import random
import myo
from collections import deque
import pickle,os
import numpy as np
from threading import Lock


cap=cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)  #width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)  #height

detector=HandDetector(maxHands=1)

timer=0
# Ean teliosi i xronometrisi tote to stateResult ginetai True
stateResult= False
start= False
scores= [0,0] #[Player, AI]

message=''
playerMove=''   # H epilogi tou paikti
choice=''       # H epilogi tou ipologisti


class Listener(myo.ApiDeviceListener):

    # def get_emg_data(self):
    #     with self.lock:
    #         return list(self.emg_data_queue)
        
    def __init__(self,buffer_len):
        super().__init__()
        self.n = buffer_len
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=buffer_len)
        self._devices = {}
        self.starter = False
        self.breaker = False

    def on_event(self, event):
        """
        Route events to the appropriate methods.
        """
        if event.type == myo.EventType.paired:
            self.on_paired(event)
        elif event.type == myo.EventType.unpaired:
            self.on_unpaired(event)
        elif event.type == myo.EventType.connected:
            self.on_connected(event)
        elif event.type == myo.EventType.pose:
            self.on_pose(event)
        elif event.type == myo.EventType.emg:
            self.on_emg(event)

    def on_paired(self, event):
        device = event.device
        self._devices[device.handle] = device
        print(f"Device paired: {event.device_name}")
        print(f"Devices tracked: {len(self._devices)}")

    def on_unpaired(self, event):
        device = event.device
        if device.handle in self._devices:
            del self._devices[device.handle]
        print("Device unpaired!")
        print(f"Devices tracked: {len(self._devices)}")

    def on_connected(self, event):
        print(f"Device connected: {event.device_name}")
        event.device.vibrate(myo.VibrationType.short)
        event.device.stream_emg(True)

    def on_pose(self, event):
        global start, stateResult, initialTime,timer
        if event.pose == myo.Pose.double_tap:
            if timer<=0:
                print("Pose detected: Start")
                self.starter = True
            
        elif event.pose == myo.Pose.wave_out:
            if timer>3 or timer<0:
                print("Pose detected: Close App")
                self.breaker = True

    def vibrate_on_result(self, result):
        """
        Trigger vibrations based on the game result.
        :param result: A string indicating 'You win' or 'You lose'.
        """
        if self._devices:
            # Use the first device in the tracked devices
            device = next(iter(self._devices.values()))
            if result == 'You won':
                device.vibrate(myo.VibrationType.short)
                device.vibrate(myo.VibrationType.short)
            elif result == 'You lost':
                device.vibrate(myo.VibrationType.long)
            elif result == "You drew":
                device.vibrate(myo.VibrationType.short)
        else:
            print("No connected devices to vibrate!")

    def on_emg(self, event):
        with self.lock:
            #print(f"EMG data received: {event.emg}")
            self.emg_data_queue.append((event.timestamp, event.emg))
    
    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    

base_dir = os.path.dirname(__file__)
myoPath = os.path.join(base_dir,"myo-sdk-win-0.9.0")
# Initialize the Myo library, by specifying the path to the SDK
myo.init(sdk_path=myoPath)
try:
    # Create a hub to manage Myo devices
    hub = myo.Hub()
except:
    print("No device")
# Create instance of the Listener class




# Parse command line inputs, if any
input_file = os.path.join(base_dir,"classification", "models","trained_model.pkl")
#print(os.path.isfile(input_file))

# Load pickled feature extractor and classification model
with open(input_file, 'rb') as file:
    model = pickle.load(file)

# Extract variables from pickled object
mdl = model['mdl']
#print(mdl)
feature_extractor = model['feature_extractor']
#print(feature_extractor)
gestures = model['gestures']

# Set up the listener in order to contain the most up-to-date readings from the MYO
listener = Listener(feature_extractor.winlen)


# Rolling window for storing recognized gestures and their timestamps
gesture_window = deque(maxlen=100)  # To store (timestamp, gesture_index) tuples

mode = 1
isMyoConnected = False

def get_result(player_gesture, pc_gesture):
    """Determine the result of Rock-Paper-Scissors."""
    rules = {
        'rock': {'rock': 'You drew', 'paper': 'You lost', 'scissors': 'You won'},
        'paper': {'rock': 'You won', 'paper': 'You drew', 'scissors': 'You lost'},
        'scissors': {'rock': 'You lost', 'paper': 'You won', 'scissors': 'You drew'}
    }
    return rules[player_gesture][pc_gesture]

while True:
    try:
        hub.run(listener.on_event, 20)            
        isMyoConnected = True
    except Exception as e:
        isMyoConnected = False
    finally:
        bgImage= cv2.imread("Resources/bg.png")
        cv2.namedWindow("BG", cv2.WINDOW_NORMAL)
        window_width, window_height = max(320,cv2.getWindowImageRect("BG")[2]), max(180,cv2.getWindowImageRect("BG")[3])
        success, img= cap.read()

        imgScaled = cv2.resize(img, None, fx=0.308, fy=0.444)
        imgScaled_resized = cv2.resize(imgScaled, (788, 852))


        # Detect Hand
        # hands, img = detector.findHands(img, draw=True, flipType=True)
        hands, img = detector.findHands(imgScaled_resized)

        cv2.putText(bgImage, 'Press', (1060, 972), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4) 
        cv2.putText(bgImage, 'S to play', (1020, 1052), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)  
        cv2.putText(bgImage, 'Press', (1060,1180), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,0), 4)
        cv2.putText(bgImage, 'Q to close', (980,1260), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,0), 4)
        if mode == 1:
            cv2.putText(bgImage, 'Enabled Mode: Optical', (820, 260), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4) 
        elif mode == 2:
            cv2.putText(bgImage, 'Enabled Mode: Haptic', (820, 260), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)  


        if mode == 1 and isMyoConnected:
            cv2.putText(bgImage, 'Press 2 for haptic mode', (1380,1360), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,0), 4)
        elif mode == 2:
            cv2.putText(bgImage, 'Press 1 for optical mode', (1380,1360), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,0), 4)


        if start:
            if stateResult is False:
                

                if mode == 1:
                    timer = time.time() - initialTime
                    cv2.putText(bgImage, str(int(3.5-timer)), (1100, 840), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 255), 16)
                    if timer > 3:
                        stateResult = True
                        timer = 0

                        # Determine AI choice
                        choices = ['paper', 'rock', 'scissors']
                        choice = choices[random.randint(0, 2)]
                        print('Computer choice:', choice)
                        
                        if hands:
                            hand = hands[0]
                            fingers = detector.fingersUp(hand)
                            playerMove = 'Move not recognised'
                            message = 'Move not recognised'

                            if fingers == [0, 0, 0, 0, 0] or fingers == [1, 0, 0, 0, 0]:
                                playerMove = "rock"
                            elif fingers == [1, 1, 1, 1, 1]:
                                playerMove = "paper"
                            elif fingers == [0, 1, 1, 0, 0]:
                                playerMove = "scissors"

                            # Determine the result
                            if playerMove != "Move not recognised":
                                message = get_result(playerMove,choice)
                                listener.vibrate_on_result(message)

                            if message == 'You won':
                                scores[0] += 1
                            elif message == "You lost":
                                scores[1] += 1

                            

                        else:
                            message = 'Move not recognised'

                        

                            # topothetisi ikonas ipologisti
                        AIimg= cv2.imread(f"Resources/{choice}.png", cv2.IMREAD_UNCHANGED)

                        new_width = 720  
                        new_height = 600  
                        AIimg_resized = cv2.resize(AIimg, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        AIimg_rgb = AIimg_resized[:, :, :3]  # First 3 channels (RGB)

                        bgImage[548:1148, 1440:2160] = AIimg_rgb       # height x width

                elif mode == 2:
                    
                        
                    #print("works")
                        

                            
                    
                    
                    
                    timer = time.time() - initialTime
                    cv2.putText(bgImage, str(int(3.5-timer)), (1100, 840), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 255), 16)
                    
                    # Collect gestures for the next 3 seconds
                    
                    if timer <= 3.0:
                                                
                        #print(timer)
                        #print(len(listener.emg_data_queue))
                        emg_data = listener.get_emg_data()
                        #print("EMG data:", emg_data)
                        #print(feature_extractor.winlen)
                        if len(listener.emg_data_queue) >= feature_extractor.winlen:
                            emg = listener.get_emg_data()
                            emg = np.array([x[1] for x in emg])
                            #print(emg)
                            feature_vector = feature_extractor.extract_feature_vector(emg)
                            inference = mdl.predict(feature_vector)
                            gesture_window.append((time.time(), inference[0]))

                    # Filter gestures from the last 3 seconds
                    if timer>3:
                        timer = 0
                        stateResult = True
                        recent_gestures = [g[1] for g in gesture_window if time.time() - g[0] <= 2.5]
                        player_gesture = max(set(recent_gestures), key=recent_gestures.count) if recent_gestures else None
                        #print(recent_gestures)

                        if player_gesture is not None:
                            playerMove = gestures[player_gesture]

                            # Generate computer's random choice
                            choice = random.choice(gestures)

                            # Determine and print the result
                            message = get_result(playerMove, choice)
                            print(f'\nYour Gesture: {playerMove}')
                            print(f'Computer Gesture: {choice}')
                            print(f'Result: {message}!\n')

                            if message == 'You won':
                                scores[0] += 1
                            elif message == "You lost":
                                scores[1] += 1

                            # Trigger vibration feedback
                            listener.vibrate_on_result(message)

                            
                            AIimg= cv2.imread(f"Resources/{choice}.png", cv2.IMREAD_UNCHANGED)
                            playerimg= cv2.imread(f"Resources/{playerMove}.png", cv2.IMREAD_UNCHANGED)

                            new_width = 720  
                            new_height = 600  
                            AIimg_resized = cv2.resize(AIimg, (new_width, new_height), interpolation=cv2.INTER_AREA)
                            AIimg_rgb = AIimg_resized[:, :, :3]  # First 3 channels (RGB)
                            playerimg_resized = cv2.resize(playerimg, (new_width, new_height), interpolation=cv2.INTER_AREA)
                            playerimg_rgb = playerimg_resized[:, :, :3]  # First 3 channels (RGB)

                            bgImage[548:1148, 1440:2160] = AIimg_rgb

                            bgImage[548:1148, 132:852] = playerimg_rgb
                        else:
                            playerMove = "No gesture detected"
                            message = "No gesture detected"
                            
                            

                            

        
        if mode == 1:
            bgImage[428:1280, 132:920] = imgScaled_resized        # height x width

        # diatirisi ikonas ipologisti
        if stateResult: 
            bgImage[548:1148, 1440:2160] = AIimg_rgb # height x width
            if mode == 2:
                bgImage[548:1148, 140:860] = playerimg_rgb        
            

        if playerMove!='' and timer==0:    

            if(playerMove=="Move not recognised"):

                cv2.putText(bgImage, 'Move not', (1020, 492), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)  
                cv2.putText(bgImage, 'recognised', (1000, 572), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)  
            
            elif(playerMove=="No gesture detected"):

                cv2.putText(bgImage, 'No gesture', (990, 492), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)  
                cv2.putText(bgImage, 'detected', (1020, 572), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4) 

            else:
                cv2.putText(bgImage, playerMove + ' vs ', (1012, 492), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)  # (253, 123) scaled
                cv2.putText(bgImage, choice, (1012, 572), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)  # (253, 143) scaled
                cv2.putText(bgImage, message, (1012, 652), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)  # (253, 163) scaled


        cv2.putText(bgImage, ':' + str(scores[0]), (420, 390), cv2.FONT_HERSHEY_PLAIN, 7, (255, 255, 255), 7)  # (105, 100) scaled
        cv2.putText(bgImage, ':' + str(scores[1]), (1508, 390), cv2.FONT_HERSHEY_PLAIN, 7, (255, 255, 255), 7)  # (377, 98) scaled


        # cv2.imshow("BG",bgImage)
        bgImage_resized = cv2.resize(bgImage, (window_width, window_height))

        # Display the resized background image
        cv2.imshow("BG", bgImage_resized)
        key= cv2.waitKey(1)

        if key == ord('s') or listener.starter:
            start= True
            initialTime=time.time()
            gesture_window.clear()

            stateResult= False
            listener.starter = False

        if key == ord('q') or listener.breaker:
            break

        # Switch mode on key press
        if key == ord('1'):
            mode = 1  # Optical mode
            start = False
            stateResult = False
            playerMove=''
        elif key == ord('2'):
            if isMyoConnected:
                mode = 2  # Haptic mode
                start = False
                stateResult = False
                playerMove=''
            else:
                print("No Myo Connected")
