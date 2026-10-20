from .database import SessionLocal, Session as SessionModel, SessionPerson, PersonMetric

class TeacherProfileService:
    def get_profile(self, teacher_id: str):
        db = SessionLocal()
        try:
            sessions = db.query(SessionModel).filter(SessionModel.teacher_id == teacher_id).order_by(SessionModel.id.desc()).all()
            if not sessions:
                return {
                    "teacher_id": teacher_id,
                    "effectiveness_score": 0.0,
                    "trend": 0.0,
                    "class_history": [],
                    "student_trends": [],
                    "strengths": ["Awaiting initial session recording"]
                }

            avg_score = sum(s.average_attention for s in sessions) / len(sessions)
            history = [
                {"name": s.session_name, "engagement": round(s.average_attention, 1)}
                for s in sessions[:5]
            ]
            
            # Real longitudinal trend between latest and earliest session
            trend = 0.0
            if len(sessions) > 1:
                trend = round(sessions[0].average_attention - sessions[-1].average_attention, 1)

            # Query real students across teacher's sessions
            session_ids = [s.id for s in sessions]
            persons = db.query(SessionPerson).filter(SessionPerson.session_id.in_(session_ids)).limit(5).all()
            student_trends = [
                {
                    "name": p.person_id,
                    "change": round(p.presence_duration, 1),
                    "status": "Attended" if p.is_present else "Partial Presence"
                }
                for p in persons
            ]

            strengths = []
            if avg_score >= 70:
                strengths.append(f"Strong overall engagement index ({avg_score:.1f}%)")
            if len(sessions) >= 3:
                strengths.append(f"Active session continuity ({len(sessions)} recorded lectures)")
            if not strengths:
                strengths.append("Session data actively tracking in SQLite")

            return {
                "teacher_id": teacher_id,
                "effectiveness_score": round(avg_score, 1),
                "trend": trend,
                "class_history": history,
                "student_trends": student_trends,
                "strengths": strengths
            }
        finally:
            db.close()
