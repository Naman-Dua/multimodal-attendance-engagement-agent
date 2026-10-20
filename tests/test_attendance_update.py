import unittest
import numpy as np
import cv2
import tempfile
import os

from backend.pipeline import ClassroomPipeline
from backend.core.database import SessionLocal, Session as SessionModel, SessionPerson

class TestAttendanceUpdate(unittest.TestCase):
    def test_attendance_records_updated_in_database(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            temp_path = tfile.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 30.0, (640, 480))
            # Generate 45 frames (~1.5 seconds) with a simulated face (skin tone HSV oval)
            for _ in range(45):
                frame = np.full((480, 640, 3), 40, dtype=np.uint8)
                # BGR skin-color ellipse
                cv2.ellipse(frame, (320, 240), (80, 110), 0, 0, 360, (140, 180, 230), -1)
                out.write(frame)
            out.release()

            pipeline = ClassroomPipeline(session_name="Live Attendance Test Session", teacher_id="PROF_TEST")
            results = pipeline.run(source=temp_path, max_frames=45)

            self.assertGreater(results["confirmed_attendees"], 0, "Attendance count should be >= 1 for visible student!")

            # Verify database rows
            db = SessionLocal()
            try:
                db_sess = db.query(SessionModel).filter(SessionModel.id == results["session_id"]).first()
                self.assertEqual(db_sess.attendance_count, results["confirmed_attendees"])
                
                person = db.query(SessionPerson).filter(SessionPerson.session_id == results["session_id"]).first()
                self.assertIsNotNone(person)
                self.assertTrue(person.is_present)
                self.assertGreater(person.presence_duration, 0.5)
            finally:
                db.close()

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    unittest.main()
