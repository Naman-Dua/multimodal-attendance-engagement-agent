import time
from typing import Dict, List, Any

class EngagementScorer:
    """
    Transforms fused multimodal signals (vision presence, movement, inactivity, 
    and audio speech activity) into interpretable temporal engagement scores.
    """
    def __init__(
        self,
        weight_presence: float = 0.40,
        weight_movement: float = 0.25,
        weight_audio: float = 0.35,
        inactivity_penalty_rate: float = 0.20,
        drop_threshold: float = 0.45,
        peak_threshold: float = 0.80,
    ):
        # Configurable, transparent weights
        self.w_presence = weight_presence
        self.w_movement = weight_movement
        self.w_audio = weight_audio
        self.inactivity_penalty_rate = inactivity_penalty_rate
        self.drop_threshold = drop_threshold
        self.peak_threshold = peak_threshold
        
        # Temporal engagement history
        self.history: List[Dict[str, Any]] = []

    def calculate_score(
        self,
        presence_ratio: float,
        movement_level: float,
        inactivity_duration_sec: float,
        speech_ratio: float,
        timestamp: float = None
    ) -> Dict[str, Any]:
        """
        Computes a normalized engagement score (0.0 to 1.0) for a given time window.

        Args:
            presence_ratio: Ratio of participants visually present (0.0 to 1.0)
            movement_level: Normalized natural head/body movement frequency (0.0 to 1.0)
            inactivity_duration_sec: Continuous seconds of participant inactivity/unresponsiveness
            speech_ratio: Proportion of time active speech was detected (0.0 to 1.0)
            timestamp: Session timestamp in seconds
        """
        ts = timestamp if timestamp is not None else time.time()

        # 1. Visual Presence Component (0.0 to 1.0)
        presence_score = min(max(presence_ratio, 0.0), 1.0)

        # 2. Movement Dynamics Component (natural motion indicates active listening)
        # Low movement (<0.1) suggests disinterest/sleeping; very high (>0.9) suggests restlessness
        if 0.2 <= movement_level <= 0.8:
            movement_score = 1.0
        elif movement_level < 0.2:
            movement_score = movement_level / 0.2
        else:
            movement_score = max(0.5, 1.0 - (movement_level - 0.8) * 2.0)

        # 3. Audio Interaction Component
        # Ideal classroom interaction has balanced speech (teacher/student dialogue)
        audio_score = min(max(speech_ratio, 0.0), 1.0)

        # 4. Inactivity Penalty (reduces engagement if inactive > 10 seconds)
        penalty = 0.0
        if inactivity_duration_sec > 10:
            penalty = min(0.30, (inactivity_duration_sec - 10) * 0.02 * self.inactivity_penalty_rate)

        # Raw fused weighted score
        raw_score = (
            (self.w_presence * presence_score) +
            (self.w_movement * movement_score) +
            (self.w_audio * audio_score) -
            penalty
        )
        final_score = round(max(0.0, min(1.0, raw_score)), 3)

        # Classify status
        if final_score < self.drop_threshold:
            status = "attention_drop"
        elif final_score >= self.peak_threshold:
            status = "peak"
        else:
            status = "stable"

        result = {
            "timestamp": round(ts, 2),
            "engagement_score": final_score,
            "status": status,
            "components": {
                "visual_presence": round(presence_score, 3),
                "movement_dynamics": round(movement_score, 3),
                "audio_speech_activity": round(audio_score, 3),
                "inactivity_penalty": round(penalty, 3),
            }
        }

        self.history.append(result)
        return result

    def get_temporal_profile(self) -> List[Dict[str, Any]]:
        """Returns the full time-series engagement profile for the session."""
        return self.history

    def get_session_summary(self) -> Dict[str, Any]:
        """Aggregates the temporal profile into overall session analytics."""
        if not self.history:
            return {
                "average_engagement": 0.0,
                "attention_drops": 0,
                "peaks": 0,
                "total_intervals": 0,
                "pacing": "uninitialized"
            }

        scores = [entry["engagement_score"] for entry in self.history]
        statuses = [entry["status"] for entry in self.history]

        drops_count = statuses.count("attention_drop")
        peaks_count = statuses.count("peak")
        avg_score = round(sum(scores) / len(scores), 3)

        if drops_count > len(scores) * 0.35:
            pacing = "pacing_slow_or_disengaged"
        elif peaks_count > len(scores) * 0.40:
            pacing = "highly_interactive"
        else:
            pacing = "steady"

        return {
            "average_engagement": avg_score,
            "min_engagement": min(scores),
            "max_engagement": max(scores),
            "attention_drops": drops_count,
            "peaks": peaks_count,
            "total_intervals": len(scores),
            "pacing": pacing
        }
