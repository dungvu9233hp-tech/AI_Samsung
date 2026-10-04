"""Âm thanh cảnh báo (pygame). Tự sinh assets/alarm.wav nếu chưa có."""
import os
import wave
import numpy as np

ALARM = os.path.join(os.path.dirname(__file__), "..", "assets", "alarm.wav")


def ensure_alarm(path=ALARM, sr=22050):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(int(sr * 0.25)) / sr
    beep = 0.6 * np.sin(2 * np.pi * 1000 * t) * (np.minimum(t, 0.25 - t) > 0.01)
    sig = np.concatenate([beep, np.zeros(int(sr * 0.15)), beep, np.zeros(int(sr * 0.35))])
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((sig * 32767).astype(np.int16).tobytes())
    return path


class AlarmPlayer:
    def __init__(self, enabled=True):
        self.snd, self.playing = None, False
        if not enabled:
            return
        try:
            import pygame
            pygame.mixer.init()
            self.snd = pygame.mixer.Sound(ensure_alarm())
        except Exception as e:
            print(f"[cảnh báo] Không bật được âm thanh: {e}")

    def set(self, on):
        if self.snd is None or on == self.playing:
            return
        self.snd.play(loops=-1) if on else self.snd.stop()
        self.playing = on


if __name__ == "__main__":
    print(ensure_alarm())
