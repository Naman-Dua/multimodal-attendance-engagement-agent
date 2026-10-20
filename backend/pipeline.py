import cv2
import numpy as np
import time
import argparse
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

from backend.core.database import SessionLocal, Session as SessionModel, SessionPerson, PersonMetric
from backend.core.face_detection import FaceDetector
from backend.core.engagement_scoring import EngagementScorer
from backend.core.recommendations_engine import RecommendationsEngine
from backend.core.ai_suggestions import AISuggestionEngine
from backend.core.analytics_service import AnalyticsService

class ClassroomPipeline:
    """
    Live real-world pipeline executing face detection, centroid tracking,
    sustained presence calculation, and multimodal engagement scoring.
    """
    def __init__(self, session_name: str = "Live Classroom Session", teacher_id: str = "PROF_01"):
        self.session_name = session_name
        self.teacher_id = teacher_id
        self.face_detector = FaceDetector()
        self.engagement_scorer = EngagementScorer()
        self.recommendations_engine = RecommendationsEngine()
        self.ai_engine = AISuggestionEngine()
        self.analytics_service = AnalyticsService()

        # Real tracking & presence state
        self.tracked_people = {}  # {track_id: {"last_centroid": (cx, cy), "first_seen": dt, "last_seen": dt, "total_sec": float}}
        self.next_track_id = 1
        self.min_presence_sec = 5.0  # Configurable sustained attendance threshold

    def _match_centroids(self, detected_bboxes: List[tuple]) -> List[int]:
        """Simple spatial centroid tracker (nearest neighbor within distance threshold)."""
        assigned_ids = []
        new_centroids = []
        for (x, y, w, h) in detected_bboxes:
            new_centroids.append((x + w // 2, y + h // 2))

        # Match each new centroid to closest existing tracked person
        used_ids = set()
        for cx, cy in new_centroids:
            matched_id = None
            min_dist = 80.0  # Maximum pixel distance threshold for matching

            for tid, tdata in self.tracked_people.items():
                if tid in used_ids:
                    continue
                ox, oy = tdata["last_centroid"]
                dist = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    matched_id = tid

            if matched_id is None:
                matched_id = f"Student_{self.next_track_id:02d}"
                self.next_track_id += 1
                self.tracked_people[matched_id] = {
                    "last_centroid": (cx, cy),
                    "first_seen": datetime.utcnow(),
                    "last_seen": datetime.utcnow(),
                    "total_sec": 0.0,
                    "prev_centroid": (cx, cy)
                }

            used_ids.add(matched_id)
            self.tracked_people[matched_id]["last_centroid"] = (cx, cy)
            self.tracked_people[matched_id]["last_seen"] = datetime.utcnow()
            assigned_ids.append(matched_id)

        return assigned_ids

    def process_single_frame(self, frame: np.ndarray, dt: float = 0.033) -> Dict[str, Any]:
        """Processes a single real video/webcam frame and returns telemetry + annotated frame."""
        detections = self.face_detector.process_frame(frame)
        bboxes = [d["bbox"] for d in detections]
        
        # Track identities across frames
        assigned_ids = self._match_centroids(bboxes)
        
        # Update presence durations
        for tid in assigned_ids:
            self.tracked_people[tid]["total_sec"] += dt

        # Calculate movement delta from centroids
        movement_sum = 0.0
        for i, (x, y, w, h) in enumerate(bboxes):
            tid = assigned_ids[i]
            tdata = self.tracked_people[tid]
            pcx, pcy = tdata.get("prev_centroid", (x, y))
            cx, cy = tdata["last_centroid"]
            dist = np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
            tdata["prev_centroid"] = (cx, cy)
            movement_sum += min(1.0, dist / 20.0)

            # Draw visual annotation on real frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, tid, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        avg_movement = (movement_sum / len(bboxes)) if bboxes else 0.0

        # Score engagement
        score_record = self.engagement_scorer.calculate_score(
            presence_ratio=min(1.0, len(bboxes) / 5.0) if bboxes else 0.0,
            movement_level=avg_movement,
            inactivity_duration_sec=0.0 if bboxes else dt,
            speech_ratio=0.35
        )

        return {
            "annotated_frame": frame,
            "attendees_count": len(bboxes),
            "assigned_ids": assigned_ids,
            "engagement_score": score_record["engagement_score"],
            "status": score_record["status"]
        }

    def run(self, source: Any = 0, max_frames: Optional[int] = None) -> Dict[str, Any]:
        """Runs the pipeline against a real video file or webcam device."""
        print(f"\n[ClassroomPipeline] Opening video source: {source}")
        
        # Handle string integer or device index
        camera_id = int(source) if isinstance(source, str) and source.isdigit() else source
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}. Ensure webcam is connected or file path is valid.")

        db = SessionLocal()
        session_record = SessionModel(
            teacher_id=self.teacher_id,
            session_name=self.session_name,
            start_time=datetime.utcnow()
        )
        db.add(session_record)
        db.commit()
        db.refresh(session_record)
        session_id = session_record.id

        frame_count = 0
        prev_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                now = time.time()
                dt = max(0.01, now - prev_time)
                prev_time = now
                frame_count += 1

                res = self.process_single_frame(frame, dt=dt)

                # Record metrics every 15 frames (~0.5s)
                if frame_count % 15 == 0:
                    for tid in res["assigned_ids"]:
                        db.add(PersonMetric(
                            session_id=session_id,
                            person_id=tid,
                            timestamp=datetime.utcnow(),
                            attention=res["engagement_score"] * 100.0,
                            emotion="focused" if res["engagement_score"] > 0.6 else "neutral"
                        ))
                    db.commit()

                if max_frames and frame_count >= max_frames:
                    break

            # Mark final attendance
            confirmed_attendees = 0
            for tid, tdata in self.tracked_people.items():
                is_present = tdata["total_sec"] >= self.min_presence_sec
                if is_present:
                    confirmed_attendees += 1
                db.add(SessionPerson(
                    session_id=session_id,
                    person_id=tid,
                    first_seen=tdata["first_seen"],
                    last_seen=tdata["last_seen"],
                    presence_duration=round(tdata["total_sec"], 2),
                    is_present=is_present
                ))

            summary = self.engagement_scorer.get_session_summary()
            session_record.end_time = datetime.utcnow()
            session_record.attendance_count = confirmed_attendees
            session_record.average_attention = summary["average_engagement"] * 100.0
            db.commit()

            # Generate real LLM suggestions
            suggestions = self.ai_engine.generate_suggestions({
                "attendance_count": confirmed_attendees,
                "average_engagement": summary["average_engagement"],
                "attention_drops": summary["attention_drops"],
                "pacing": summary["pacing"]
            })

            return {
                "session_id": session_id,
                "frames_processed": frame_count,
                "confirmed_attendees": confirmed_attendees,
                "summary": summary,
                "suggestions": suggestions
            }
        finally:
            cap.release()
            db.close()
