import numpy as np

class AudioAnalyzer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def analyze_audio_chunk(self, audio_data):
        """
        Analyzes raw audio bytes/array: noise level (dB), speech vs silence ratio.
        """
        try:
            if isinstance(audio_data, bytes):
                y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                y = np.asarray(audio_data, dtype=np.float32)

            if len(y) == 0:
                return None

            rms = float(np.sqrt(np.mean(y ** 2)))
            noise_db = float(20 * np.log10(rms + 1e-6))

            # Chunk into 25ms windows with stdlib numpy (no librosa/scipy dependency)
            chunk_size = int(self.sample_rate * 0.025)
            num_chunks = len(y) // chunk_size
            if num_chunks > 0:
                chunks = y[:num_chunks * chunk_size].reshape(num_chunks, chunk_size)
                frame_rms = np.sqrt(np.mean(chunks ** 2, axis=1))
                speech_threshold = max(0.01, float(np.mean(frame_rms) * 1.2))
                speech_ratio = float(np.sum(frame_rms > speech_threshold) / num_chunks)
            else:
                speech_ratio = 1.0 if rms > 0.02 else 0.0

            audio_level = "silent" if noise_db < -45 else ("moderate" if noise_db < -25 else "noisy")

            return {
                "rms": round(rms, 4),
                "noise_level_db": round(noise_db, 2),
                "speech_ratio": round(speech_ratio, 3),
                "audio_level": audio_level,
                "is_speech_active": speech_ratio > 0.3
            }
        except Exception as e:
            return {
                "rms": 0.0,
                "noise_level_db": -60.0,
                "speech_ratio": 0.0,
                "audio_level": "silent",
                "is_speech_active": False,
                "error": str(e)
            }
