import streamlit as st
import cv2
import tempfile
import time
import os
import pandas as pd
from datetime import datetime

from backend.core.database import SessionLocal, Session as SessionModel, SessionPerson, PersonMetric
from backend.core.analytics_service import AnalyticsService
from backend.core.teacher_profiles import TeacherProfileService
from backend.core.ai_suggestions import AISuggestionEngine
from backend.core.recommendations_engine import RecommendationsEngine
from backend.pipeline import ClassroomPipeline

st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide"
)

# Services
analytics_service = AnalyticsService()
teacher_service = TeacherProfileService()
ai_engine = AISuggestionEngine()
recs_engine = RecommendationsEngine()

st.title("🎓 Multimodal Attendance & Engagement Agent")
st.caption("Live AI Classroom Intelligence — Real-Time Face Tracking, Presence Duration & Multimodal Analytics")

# Sidebar: Controls & Live Source Ingestion
st.sidebar.header("Classroom Ingestion Mode")

input_mode = st.sidebar.radio(
    "Select Video Source:",
    ["📹 Live Webcam", "📁 Upload Video File", "📊 View Existing Sessions"]
)

# ----------------- Mode 1: Live Webcam -----------------
if input_mode == "📹 Live Webcam":
    st.subheader("📹 Live Classroom Webcam Ingestion")
    session_title = st.text_input("Session Title:", "Real-Time Classroom Lecture")
    teacher_name = st.text_input("Instructor ID:", "PROF_01")
    
    col_btn1, col_btn2 = st.columns([1, 4])
    run_live = col_btn1.button("Start Live Capture")
    stop_live = col_btn2.button("Stop Capture")

    if run_live:
        st.info("Streaming from local webcam (Device 0). Press 'Stop Capture' or uncheck to conclude.")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not access local webcam (Device 0). Please check camera permissions or try uploading a video.")
        else:
            pipeline = ClassroomPipeline(session_name=session_title, teacher_id=teacher_name)
            
            # Start DB session
            db = SessionLocal()
            session_rec = SessionModel(
                teacher_id=teacher_name,
                session_name=session_title,
                start_time=datetime.utcnow()
            )
            db.add(session_rec)
            db.commit()
            db.refresh(session_rec)
            session_id = session_rec.id
            db.close()

            frame_window = st.image([])
            metric_cols = st.columns(3)
            m1 = metric_cols[0].empty()
            m2 = metric_cols[1].empty()
            m3 = metric_cols[2].empty()

            prev_t = time.time()
            frame_idx = 0
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    now = time.time()
                    dt = max(0.01, now - prev_t)
                    prev_t = now
                    frame_idx += 1

                    telemetry = pipeline.process_single_frame(frame, dt=dt)
                    
                    # Convert BGR to RGB for Streamlit rendering
                    rgb_frame = cv2.cvtColor(telemetry["annotated_frame"], cv2.COLOR_BGR2RGB)
                    frame_window.image(rgb_frame, channels="RGB")

                    # Live metrics
                    m1.metric("Participants in View", f"{telemetry['attendees_count']}")
                    m2.metric("Engagement Score", f"{telemetry['engagement_score']*100:.1f}%")
                    m3.metric("State", telemetry["status"].upper())

                    # Record to DB every 30 frames (~1 sec)
                    if frame_idx % 30 == 0:
                        db = SessionLocal()
                        for tid in telemetry["assigned_ids"]:
                            db.add(PersonMetric(
                                session_id=session_id,
                                person_id=tid,
                                timestamp=datetime.utcnow(),
                                attention=telemetry["engagement_score"] * 100.0,
                                emotion="focused" if telemetry["engagement_score"] > 0.6 else "neutral"
                            ))
                        db.commit()
                        db.close()

                    if frame_idx >= 300:  # Safety cap at 300 frames (~10s per burst)
                        break

            finally:
                cap.release()
                # Finalize session in DB
                db = SessionLocal()
                sess = db.query(SessionModel).filter(SessionModel.id == session_id).first()
                if sess:
                    sess.end_time = datetime.utcnow()
                    confirmed = 0
                    for tid, tdata in pipeline.tracked_people.items():
                        is_p = tdata["total_sec"] >= pipeline.min_presence_sec
                        if is_p:
                            confirmed += 1
                        db.add(SessionPerson(
                            session_id=session_id,
                            person_id=tid,
                            first_seen=tdata["first_seen"],
                            last_seen=tdata["last_seen"],
                            presence_duration=round(tdata["total_sec"], 2),
                            is_present=is_p
                        ))
                    summary = pipeline.engagement_scorer.get_session_summary()
                    sess.attendance_count = confirmed
                    sess.average_attention = summary["average_engagement"] * 100.0
                    db.commit()
                db.close()
                st.success(f"Session #{session_id} finalized and saved to SQLite database!")

# ----------------- Mode 2: Upload Video File -----------------
elif input_mode == "📁 Upload Video File":
    st.subheader("📁 Process Pre-Recorded Classroom Video")
    uploaded_file = st.file_uploader("Upload Classroom Video (.mp4, .avi, .mov):", type=["mp4", "avi", "mov"])
    v_session_title = st.text_input("Lecture Name:", "Recorded Lecture Analysis")
    v_teacher = st.text_input("Teacher ID:", "PROF_01")

    if uploaded_file and st.button("Process Video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            temp_path = tfile.name

        try:
            st.info("Processing video frames through Vision & Audio Pipeline...")
            progress_bar = st.progress(0.0)
            preview_placeholder = st.empty()

            pipeline = ClassroomPipeline(session_name=v_session_title, teacher_id=v_teacher)
            cap = cv2.VideoCapture(temp_path)
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100

            db = SessionLocal()
            session_rec = SessionModel(
                teacher_id=v_teacher,
                session_name=v_session_title,
                start_time=datetime.utcnow()
            )
            db.add(session_rec)
            db.commit()
            db.refresh(session_rec)
            session_id = session_rec.id
            db.close()

            curr_f = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                curr_f += 1
                telemetry = pipeline.process_single_frame(frame, dt=0.033)
                
                # Update UI periodically
                if curr_f % 15 == 0:
                    progress_bar.progress(min(1.0, curr_f / total_f))
                    rgb = cv2.cvtColor(telemetry["annotated_frame"], cv2.COLOR_BGR2RGB)
                    preview_placeholder.image(rgb, caption=f"Frame {curr_f}/{total_f} - Attendees: {telemetry['attendees_count']}", width="stretch")

            cap.release()
            
            # Finalize
            db = SessionLocal()
            sess = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            if sess:
                sess.end_time = datetime.utcnow()
                confirmed = 0
                for tid, tdata in pipeline.tracked_people.items():
                    is_p = tdata["total_sec"] >= pipeline.min_presence_sec
                    if is_p:
                        confirmed += 1
                    db.add(SessionPerson(
                        session_id=session_id,
                        person_id=tid,
                        first_seen=tdata["first_seen"],
                        last_seen=tdata["last_seen"],
                        presence_duration=round(tdata["total_sec"], 2),
                        is_present=is_p
                    ))
                summary = pipeline.engagement_scorer.get_session_summary()
                sess.attendance_count = confirmed
                sess.average_attention = summary["average_engagement"] * 100.0
                db.commit()
            db.close()

            st.success(f"Processing complete! Session #{session_id} saved.")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# ----------------- Mode 3: View Existing Sessions -----------------
st.markdown("---")
db = SessionLocal()
all_sessions = []
try:
    all_sessions = db.query(SessionModel).order_by(SessionModel.id.desc()).all()
finally:
    db.close()

if not all_sessions:
    st.info("No classroom sessions available yet. Choose 'Live Webcam' or 'Upload Video File' in the sidebar to create your first session!")
else:
    st.subheader("📊 Session Analytics & Attendance Telemetry")
    sess_map = {f"Session #{s.id} - {s.session_name} ({s.start_time.strftime('%Y-%m-%d %H:%M')})": s.id for s in all_sessions}
    chosen_label = st.selectbox("Select Session to Inspect:", list(sess_map.keys()))
    sel_id = sess_map[chosen_label]

    db = SessionLocal()
    cur_sess = None
    cur_persons = []
    try:
        cur_sess = db.query(SessionModel).filter(SessionModel.id == sel_id).first()
        cur_persons = db.query(SessionPerson).filter(SessionPerson.session_id == sel_id).all()
    finally:
        db.close()

    # KPI Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confirmed Attendance", f"{cur_sess.attendance_count if cur_sess else 0} Students")
    c2.metric("Mean Attention Index", f"{cur_sess.average_attention if cur_sess else 0.0:.1f}%")
    c3.metric("Assigned Instructor", cur_sess.teacher_id if cur_sess else "N/A")
    c4.metric("Status", "Completed" if cur_sess and cur_sess.end_time else "Active")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Attendance Register", 
        "📈 Attention Timeline", 
        "💡 AI Teaching Suggestions", 
        "👤 Individual Student Analytics", 
        "👩‍🏫 Teacher Profile"
    ])

    with tab1:
        st.subheader("Presence-Based Attendance Records")
        if cur_persons:
            p_data = [{
                "Student ID": p.person_id,
                "First Observed": p.first_seen.strftime("%H:%M:%S") if p.first_seen else "N/A",
                "Last Observed": p.last_seen.strftime("%H:%M:%S") if p.last_seen else "N/A",
                "Presence Duration (s)": round(p.presence_duration, 1),
                "Attendance Verdict": "✅ Attended" if p.is_present else "❌ Insufficient Presence"
            } for p in cur_persons]
            st.dataframe(pd.DataFrame(p_data), width="stretch")
        else:
            st.info("No participant tracks recorded for this session.")

    with tab2:
        st.subheader("Session Engagement Timeline")
        trends = analytics_service.get_session_trends(sel_id)
        if trends.get("timeline"):
            df_t = pd.DataFrame(trends["timeline"]).set_index("timestamp")[["attention"]]
            st.line_chart(df_t, width="stretch")
        else:
            st.info("No telemetry points available for this session.")

    with tab3:
        st.subheader("Pedagogical Decision Support (AI Engine)")
        ai_tips = ai_engine.generate_suggestions({
            "attendance_count": cur_sess.attendance_count if cur_sess else 0,
            "average_engagement": (cur_sess.average_attention / 100.0) if cur_sess else 0.7,
            "audio_level": "moderate"
        })
        for tip in ai_tips.get("suggestions", []):
            st.markdown(f"- 🎯 **{tip}**")

    with tab4:
        st.subheader("Student Self-Reflection Telemetry")
        if cur_persons:
            chosen_pid = st.selectbox("Inspect Student ID:", [p.person_id for p in cur_persons])
            st_data = analytics_service.get_student_session_analytics(sel_id, chosen_pid)
            
            s1, s2, s3 = st.columns(3)
            s1.metric("Confirmed Present", "Yes" if st_data.get("attendance_confirmed") else "No")
            s2.metric("Presence Time", f"{st_data.get('presence_duration_seconds', 0)}s")
            s3.metric("Focus Consistency", f"{st_data.get('focus_consistency_percentage', 0)}%")
        else:
            st.info("No student telemetry recorded.")

    with tab5:
        st.subheader("Teacher Effectiveness Scorecard")
        prof = teacher_service.get_profile(cur_sess.teacher_id if cur_sess else "PROF_01")
        st.metric("Instructional Effectiveness Score", f"{prof['effectiveness_score']}%", f"{prof['trend']}% Trend")
        st.markdown("##### Identified Strengths")
        for s in prof.get("strengths", []):
            st.markdown(f"- ⭐ {s}")
        st.markdown("##### Recent Session History")
        st.dataframe(pd.DataFrame(prof.get("class_history", [])), width="stretch")
