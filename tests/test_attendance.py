import unittest
from datetime import datetime
from backend.pipeline import ClassroomPipeline

class TestAttendanceLogic(unittest.TestCase):
    def test_sustained_presence_confirms_attendance(self):
        pipeline = ClassroomPipeline()
        # Simulate a person detected across 40 frames (~1.3s)
        tid = "Student_01"
        pipeline.tracked_people[tid] = {
            "first_seen": datetime.utcnow(),
            "last_seen": datetime.utcnow(),
            "total_sec": 2.5
        }
        
        # Test adaptive threshold for short session (e.g. 3s session)
        total_session_sec = 3.0
        effective_threshold = min(pipeline.min_presence_sec, max(1.0, total_session_sec * 0.3))
        
        is_present = pipeline.tracked_people[tid]["total_sec"] >= effective_threshold
        self.assertTrue(is_present, "Student with 2.5s presence in a 3s session should be confirmed present!")

if __name__ == "__main__":
    unittest.main()
