import unittest
import numpy as np
import cv2
import tempfile
import os

from backend.pipeline import ClassroomPipeline
from backend.core.database import SessionLocal, Session as SessionModel

class TestClassroomPipeline(unittest.TestCase):
    def test_pipeline_run_on_video_file(self):
        # Create a small real 10-frame temporary video file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            temp_video_path = tfile.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (320, 240))
            for _ in range(10):
                frame = np.full((240, 320, 3), 128, dtype=np.uint8)
                out.write(frame)
            out.release()

            # Execute pipeline
            pipeline = ClassroomPipeline(session_name="TDD Test Video Session", teacher_id="TDD_PROF")
            res = pipeline.run(source=temp_video_path, max_frames=10)

            self.assertIn("session_id", res)
            self.assertIn("confirmed_attendees", res)
            self.assertIn("summary", res)
            self.assertEqual(res["frames_processed"], 10)

            # Verify session was recorded in database
            db = SessionLocal()
            try:
                db_session = db.query(SessionModel).filter(SessionModel.id == res["session_id"]).first()
                self.assertIsNotNone(db_session)
                self.assertEqual(db_session.teacher_id, "TDD_PROF")
            finally:
                db.close()

        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

if __name__ == "__main__":
    unittest.main()
