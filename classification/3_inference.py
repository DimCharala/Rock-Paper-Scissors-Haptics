from collections import deque
from collections import Counter
import pickle, time, sys, os
import numpy as np
import random
from sklearn import svm
import myo
import keyboard  # For listening to key presses
from myo_ecn.listeners import Buffer, ConnectionChecker
from EMG_Classification import FeatureExtractor, ClassificationModel

class Responder(Buffer):
          

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    def vibrate_on_result(self, result):
        """
        Trigger vibrations based on the game result.
        :param result: A string indicating 'Win' or 'Lose'.
        """
        if self._devices:
            # Use the first device in the tracked devices
            device = next(iter(self._devices.values()))
            if result == 'Win':
                device.vibrate(myo.VibrationType.medium)
            elif result == 'Lose':
                device.vibrate(myo.VibrationType.short)
            elif result == 'Draw':
                device.vibrate(myo.VibrationType.long)
        else:
            print("No connected devices to vibrate!")

    def on_event(self, event):
        """
        Route events to appropriate methods.
        """
        if event.type == myo.EventType.connected:
            self.on_connected(event)
        elif event.type == myo.EventType.unpaired:
            self.on_unpaired(event)
        elif event.type == myo.EventType.emg:
            self.on_emg(event)

    def on_connected(self, event):
        device = event.device
        self._devices[device.handle] = device
        event.device.stream_emg(True)
        print(f"Device connected: {event.device_name}. Devices tracked: {len(self._devices)}")

    def on_unpaired(self, event):
        device = event.device
        if device.handle in self._devices:
            del self._devices[device.handle]
        print(f"Device unpaired! Devices tracked: {len(self._devices)}")

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))


def get_result(player_gesture, pc_gesture):
    """Determine the result of Rock-Paper-Scissors."""
    rules = {
        'rock': {'rock': 'Draw', 'paper': 'Lose', 'scissors': 'Win'},
        'paper': {'rock': 'Win', 'paper': 'Draw', 'scissors': 'Lose'},
        'scissors': {'rock': 'Lose', 'paper': 'Win', 'scissors': 'Draw'}
    }
    return rules[player_gesture][pc_gesture]


def main():

    base_dir = os.path.dirname(os.path.dirname(__file__))  # Get script's directory
    
    myo_folder = os.path.join(base_dir, 'myo-sdk-win-0.9.0')

    # ================== setup myo-python (do not change) =====================
    myo.init(sdk_path=myo_folder)  # Compile Python binding to Myo's API
    hub = myo.Hub()  # Create a Python instance of MYO API
    if not ConnectionChecker().ok:  # Check connection before starting acquisition:
        quit()
    # =========================================================================
    
    

    # Parse command line inputs, if any
    input_file = os.path.join(base_dir, 'classification', 'models', 'trained_model.pkl')
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    # Load pickled feature extractor and classification model
    with open(input_file, 'rb') as file:
        model = pickle.load(file)

    # Extract variables from pickled object
    mdl = model['mdl']
    feature_extractor = model['feature_extractor']
    gestures = model['gestures']

    # Set up the buffer that will always contain the most up-to-date readings from the MYO
    emg_buffer = Responder(feature_extractor.winlen)

    # Rolling window for storing recognized gestures and their timestamps
    gesture_window = deque(maxlen=100)  # To store (timestamp, gesture_index) tuples

    # Set up inference
    
    print('Press "s" to start the game. Perform gestures after the countdown.')

    while True:
        hub.run(emg_buffer.on_event,10)
        time.sleep(0.050)
        
        if keyboard.is_pressed('s'):
            print('\nGet ready! Game starting in...')
            for i in range(3, 0, -1):
                print(f'{i}...', end='', flush=True)
                time.sleep(1)
            gesture_window.clear()
            print('\nStart your gesture now!')

            # Collect gestures for the next 3 seconds
            start_time = time.time()
            while time.time() - start_time <= 3.0:
                if len(emg_buffer.emg_data_queue) >= feature_extractor.winlen:
                    emg = emg_buffer.get_emg_data()
                    emg = np.array([x[1] for x in emg])
                    #print(emg)
                    feature_vector = feature_extractor.extract_feature_vector(emg)
                    inference = mdl.predict(feature_vector)
                    gesture_window.append((time.time(), inference[0]))

            # Filter gestures from the last 3 seconds
            recent_gestures = [g[1] for g in gesture_window if time.time() - g[0] <= 2.5]
            player_gesture = max(set(recent_gestures), key=recent_gestures.count) if recent_gestures else None

            gesture_counts = Counter(recent_gestures)
            print("Gesture Counts:", dict(gesture_counts))
            
            

            if player_gesture is not None:
                player_gesture_name = gestures[player_gesture]
            else:
                print('No gesture detected. Try again!')
                continue

            # Generate computer's random choice
            pc_gesture_name = random.choice(gestures)

            # Determine and print the result
            result = get_result(player_gesture_name, pc_gesture_name)
            print(f'\nYour Gesture: {player_gesture_name}')
            print(f'Computer Gesture: {pc_gesture_name}')
            print(f'Result: You {result}!\n')

            # Trigger vibration feedback
            emg_buffer.vibrate_on_result(result)

            # Wait for the next game
            print('Press "s" to play again or Ctrl-C to exit.')


if __name__ == '__main__':
    main()
