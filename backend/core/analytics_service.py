from sqlalchemy.orm import Session
from sqlalchemy import func
from .database import SessionLocal, Session as SessionModel, SessionPerson, PersonMetric

class AnalyticsService:
    def get_session_trends(self, session_id: int):
        db = SessionLocal()
        try:
            # Aggregate attention over time (minute buckets)
            # SQLite specific time grouping
            # For simplicity, we just return all metrics and let frontend downsample or bin

            # Fetch last 1000 points to avoid overload
            metrics = db.query(PersonMetric)\
                .filter(PersonMetric.session_id == session_id)\
                .order_by(PersonMetric.timestamp.asc())\
                .all()

            # Group by timestamp (seconds)
            timeline = {}
            for m in metrics:
                ts_str = m.timestamp.strftime("%H:%M:%S")
                if ts_str not in timeline:
                    timeline[ts_str] = {'att': [], 'emo': {}}
                timeline[ts_str]['att'].append(m.attention)
                if m.emotion:
                    timeline[ts_str]['emo'][m.emotion] = timeline[ts_str]['emo'].get(m.emotion, 0) + 1

            # Format timeline for chart
            formatted_timeline = []
            for ts, data in timeline.items():
                avg_attention = sum(data['att']) / len(data['att']) if data['att'] else 0
                formatted_timeline.append({
                    "timestamp": ts,
                    "attention": round(avg_attention, 1),
                    "emotions": data['emo']
                })

            return {
                "session_id": session_id,
                "timeline": formatted_timeline
            }
        except Exception as e:
            print(f"Error fetching session trends: {e}")
            return {
                "session_id": session_id,
                "timeline": []
            }
        finally:
            db.close()

    def get_student_session_analytics(self, session_id: int, person_id: str):
        """
        Generates student-facing reflective analytics (attendance duration,
        engagement consistency, and focus timelines) without peer comparisons.
        """
        db = SessionLocal()
        try:
            person = db.query(SessionPerson)\
                .filter(SessionPerson.session_id == session_id, SessionPerson.person_id == person_id)\
                .first()

            metrics = db.query(PersonMetric)\
                .filter(PersonMetric.session_id == session_id, PersonMetric.person_id == person_id)\
                .order_by(PersonMetric.timestamp.asc())\
                .all()

            attention_scores = [m.attention for m in metrics] if metrics else []
            avg_att = round(sum(attention_scores) / len(attention_scores), 1) if attention_scores else 0.0

            # Calculate engagement consistency (percentage of time attention was stable >= 60%)
            high_focus_count = sum(1 for a in attention_scores if a >= 60)
            consistency_ratio = round((high_focus_count / len(attention_scores)) * 100, 1) if attention_scores else 0.0

            return {
                "student_id": person_id,
                "session_id": session_id,
                "attendance_confirmed": person.is_present if person else False,
                "presence_duration_seconds": round(person.presence_duration, 1) if person else 0.0,
                "average_attention": avg_att,
                "focus_consistency_percentage": consistency_ratio,
                "timeline": [
                    {"time": m.timestamp.strftime("%H:%M:%S"), "attention": m.attention}
                    for m in metrics
                ]
            }
        except Exception as e:
            print(f"Error generating student analytics: {e}")
            return {
                "student_id": person_id,
                "error": str(e)
            }
        finally:
            db.close()
