from pynput import keyboard
import cv2
from deepface import DeepFace
import numpy as np

CAMERA_ID = 0

data = np.recfromcsv('data.csv', delimiter=',')
columns = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
keys = [d[0] for d in data]
values = [np.array(list(d)[1:]) for d in data]


def run():
    face_analysis = None
    camera = cv2.VideoCapture(CAMERA_ID)
    i = 0
    while face_analysis is None and i < 10:
        i += 1
        try:
            _, image = camera.read()
            face_analysis = DeepFace.analyze(
                img_path=image,
                actions=['emotion'],
                prog_bar=False
            )
        except:
            pass

    camera.release()

    if face_analysis:
        vector = np.array([face_analysis['emotion'][col] * 0.01 for col in columns])
        distances = np.linalg.norm(values - vector, axis=1)
        output = keys[np.argsort(distances)[0]]

        controller = keyboard.Controller()
        controller.press(output)
    else:
        print("error")


def for_canonical(f):
    return lambda k: f(l.canonical(k))


hotkey = keyboard.HotKey(keyboard.HotKey.parse('<ctrl>+<alt>+f'), run)
with keyboard.Listener(on_press=for_canonical(hotkey.press), on_release=for_canonical(hotkey.release)) as l:
    l.join()

