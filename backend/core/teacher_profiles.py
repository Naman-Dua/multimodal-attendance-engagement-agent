from .database import SessionLocal, Session as SessionModel

class TeacherProfileService:
    def get_profile(self, teacher_id: str):
        # Mock calculation - in production, aggregate real session data
        # Real logic: fetch average attention from all sessions by this teacher

        return {
            "teacher_id": teacher_id,
            "effectiveness_score": 87.5,
            "trend": 12.0, # +12%
            "class_history": [
                {"name": "CS101 (Today)", "engagement": 85},
                {"name": "CS101 (Yesterday)", "engagement": 78},
                {"name": "Math Workshop", "engagement": 92}
            ],
            "student_trends": [
                {"name": "Priya", "change": 15, "status": "improved"},
                {"name": "Kumar", "change": 8, "status": "improved"},
                {"name": "Asha", "change": -5, "status": "declining"}
            ],
            "strengths": [
                "High engagement sessions (85%+)",
                "Emotion diversity management"
            ]
        }
