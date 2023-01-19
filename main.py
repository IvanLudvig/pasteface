from pynput import keyboard
import cv2
from deepface import DeepFace
import math


def run():
    face_analysis = None
    camera = cv2.VideoCapture(0)
    while face_analysis is None:
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
    predicted_emoji = {
        'angry': ['😠', '😡'],
        'disgust': ['🤢', '🤮'],
        'fear': ['😧', '😨', '😱'],
        'happy': ['🙂', '😄', '😆'],
        'sad': ['🙁', '😞', '😢'],
        'surprise': ['😮', '🤯'],
        'neutral': ['😐']
    }

    dominant_emotion = face_analysis['dominant_emotion']
    scale = face_analysis['emotion'][dominant_emotion]
    predicted_emoji_list = predicted_emoji[dominant_emotion]

    def clamp(x):
        return max(min(x, len(predicted_emoji_list) - 1), 0)
    idx = clamp(math.ceil((len(predicted_emoji_list) - 1) * (scale - 50) / 50))

    controller = keyboard.Controller()
    controller.press(predicted_emoji_list[idx])


def for_canonical(f):
    return lambda k: f(l.canonical(k))


hotkey = keyboard.HotKey(keyboard.HotKey.parse('<ctrl>+<alt>+f'), run)
with keyboard.Listener(on_press=for_canonical(hotkey.press), on_release=for_canonical(hotkey.release)) as l:
    l.join()
