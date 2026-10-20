from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import asyncio

from backend.core.database import SessionLocal, Session as SessionModel, SessionPerson, PersonMetric
from backend.core.face_detection import FaceDetector
from backend.core.camera_service import CameraService
from backend.core.audio_analysis import AudioAnalyzer
from backend.core.engagement_scoring import EngagementScorer
from backend.core.recommendations_engine import RecommendationsEngine
from backend.core.ai_suggestions import AISuggestionEngine
from backend.core.analytics_service import AnalyticsService
from backend.core.teacher_profiles import TeacherProfileService

app = FastAPI(
    title="Multimodal Attendance & Engagement Agent API",
    description="Privacy-preserving classroom intelligence platform integrating vision, audio, and agent reasoning.",
    version="1.0.0"
)

# Enable CORS for local frontend/dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core service singletons
face_detector = FaceDetector()
audio_analyzer = AudioAnalyzer()
engagement_scorer = EngagementScorer()
recommendations_engine = RecommendationsEngine()
ai_suggestion_engine = AISuggestionEngine()
analytics_service = AnalyticsService()
teacher_service = TeacherProfileService()

# In-memory active session tracking
active_session_state: Dict[str, Any] = {
    "session_id": None,
    "is_active": False,
    "start_time": None,
    "active_people": {}
}

class StartSessionRequest(BaseModel):
    teacher_id: str
    session_name: str

class SessionSummaryResponse(BaseModel):
    session_id: int
    session_name: str
    teacher_id: str
    duration_seconds: float
    attendance_count: int
    average_engagement: float
    recommendations: List[Dict[str, Any]]

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Multimodal Attendance & Engagement Agent",
        "active_session": active_session_state["session_id"]
    }

@app.post("/api/sessions/start")
def start_session(req: StartSessionRequest):
    db = SessionLocal()
    try:
        new_session = SessionModel(
            teacher_id=req.teacher_id,
            session_name=req.session_name,
            start_time=datetime.utcnow()
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)

        active_session_state["session_id"] = new_session.id
        active_session_state["is_active"] = True
        active_session_state["start_time"] = datetime.utcnow()
        active_session_state["active_people"] = {}

        return {
            "status": "started",
            "session_id": new_session.id,
            "session_name": new_session.session_name,
            "teacher_id": new_session.teacher_id
        }
    finally:
        db.close()

@app.post("/api/sessions/{session_id}/stop")
def stop_session(session_id: int):
    db = SessionLocal()
    try:
        session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        session.end_time = datetime.utcnow()
        summary = engagement_scorer.get_session_summary()
        session.average_attention = summary.get("average_engagement", 0.0) * 100
        
        # Calculate final attendance count
        attendance_count = db.query(SessionPerson).filter(
            SessionPerson.session_id == session_id,
            SessionPerson.is_present == True
        ).count()
        session.attendance_count = attendance_count

        db.commit()

        # Generate AI teaching suggestions for the session
        ai_tips = ai_suggestion_engine.generate_suggestions({
            "attendance_count": attendance_count,
            "average_engagement": summary.get("average_engagement", 0.7),
            "attention_drops": summary.get("attention_drops", 0),
            "pacing": summary.get("pacing", "steady")
        })

        active_session_state["is_active"] = False
        active_session_state["session_id"] = None

        return {
            "status": "stopped",
            "session_id": session.id,
            "attendance_count": attendance_count,
            "session_summary": summary,
            "ai_teaching_suggestions": ai_tips
        }
    finally:
        db.close()

@app.get("/api/sessions/{session_id}/trends")
def get_trends(session_id: int):
    return analytics_service.get_session_trends(session_id)

@app.get("/api/sessions/{session_id}/student/{student_id}")
def get_student_report(session_id: int, student_id: str):
    return analytics_service.get_student_session_analytics(session_id, student_id)

@app.get("/api/teachers/{teacher_id}/profile")
def get_teacher_profile(teacher_id: str):
    return teacher_service.get_profile(teacher_id)

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    Receives real-time streaming PCM audio bytes from AudioCapture.jsx,
    computes acoustic metrics, and pushes real-time engagement updates.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            metrics = audio_analyzer.analyze_audio_chunk(data)
            if metrics:
                # Update engagement scorer with speech ratio
                score_data = engagement_scorer.calculate_score(
                    presence_ratio=1.0 if active_session_state.get("active_people") else 0.5,
                    movement_level=0.5,
                    inactivity_duration_sec=0.0,
                    speech_ratio=metrics["speech_ratio"]
                )
                await websocket.send_json({
                    "audio_metrics": metrics,
                    "realtime_engagement": score_data
                })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
