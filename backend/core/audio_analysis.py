import librosa
import numpy as np
import os

class AudioAnalyzer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def analyze_audio_chunk(self, audio_data):
        """
        Analyzes raw audio bytes/array:
        1. Noise level (decibels)
        2. Speech vs silence ratio
        3. Sound activity patterns

        Returns engagement metrics
        """
        try:
            # Convert bytes to float32 numpy array if needed
            if isinstance(audio_data, bytes):
                # Assuming 16-bit PCM
                y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                y = audio_data

            if len(y) == 0:
                return None

            # Calculate RMS energy
            rms = float(np.sqrt(np.mean(y ** 2)))
            noise_db = float(20 * np.log10(rms + 1e-6))

            # Speech vs silence detection based on frame energy
            frame_length = int(self.sample_rate * 0.025)  # 25ms
            hop_length = int(self.sample_rate * 0.010)    # 10ms
            
            if len(y) >= frame_length:
                frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
                frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
                speech_threshold = max(0.01, float(np.mean(frame_rms) * 1.2))
                active_frames = np.sum(frame_rms > speech_threshold)
                speech_ratio = float(active_frames / len(frame_rms)) if len(frame_rms) > 0 else 0.0
            else:
                speech_ratio = 1.0 if rms > 0.02 else 0.0

            # Determine qualitative audio level
            if noise_db < -45:
                audio_level = "silent"
            elif noise_db < -25:
                audio_level = "moderate"
            else:
                audio_level = "noisy"

            return {
                "rms": round(rms, 4),
                "noise_level_db": round(noise_db, 2),
                "speech_ratio": round(speech_ratio, 3),
                "audio_level": audio_level,
                "is_speech_active": speech_ratio > 0.3
            }

        except Exception as e:
            print(f"Audio analysis error: {e}")
            return {
                "rms": 0.0,
                "noise_level_db": -60.0,
                "speech_ratio": 0.0,
                "audio_level": "silent",
                "is_speech_active": False,
                "error": str(e)
            }
