import streamlit as st
import tempfile
import os
import pandas as pd

from backend.core.database import SessionLocal, Session as SessionModel, SessionPerson, PersonMetric
from backend.core.analytics_service import AnalyticsService
from backend.core.teacher_profiles import TeacherProfileService
from backend.core.ai_suggestions import AISuggestionEngine
from backend.pipeline import ClassroomPipeline

st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide"
)

# Core services
analytics_service = AnalyticsService()
teacher_service = TeacherProfileService()
ai_engine = AISuggestionEngine()

st.title("🎓 Multimodal Attendance & Engagement Agent")
st.caption("AI Classroom Intelligence — Face Tracking, Presence Duration & Multimodal Analytics")

# Sidebar: Unified Source Ingestion
st.sidebar.header("Classroom Ingestion")
input_mode = st.sidebar.radio(
    "Select Video Source:",
    ["📹 Live Webcam", "📁 Upload Video File", "📊 View Existing Sessions"]
)

# ----------------- Mode 1: Live Webcam -----------------
if input_mode == "📹 Live Webcam":
    st.subheader("📹 Live Classroom Webcam Ingestion")
    session_title = st.text_input("Session Title:", "Live Classroom Lecture")
    teacher_name = st.text_input("Instructor ID:", "PROF_01")
    frames_limit = st.slider("Capture Duration (Frames):", min_value=30, max_value=300, value=90, step=30)

    if st.button("Start Live Capture & Attendance"):
        with st.spinner("Processing live webcam stream..."):
            try:
                pipeline = ClassroomPipeline(session_name=session_title, teacher_id=teacher_name)
                res = pipeline.run(source=0, max_frames=frames_limit)
                st.success(f"Session #{res['session_id']} completed! {res['confirmed_attendees']} attendees confirmed.")
                st.rerun()
            except Exception as e:
                st.error(f"Error accessing camera device 0: {e}")

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
            with st.spinner("Processing video frames and computing engagement metrics..."):
                pipeline = ClassroomPipeline(session_name=v_session_title, teacher_id=v_teacher)
                res = pipeline.run(source=temp_path)
                st.success(f"Session #{res['session_id']} processed! {res['confirmed_attendees']} attendees confirmed.")
                st.rerun()
        except Exception as e:
            st.error(f"Error processing video: {e}")
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
    st.info("No classroom sessions available yet. Choose 'Live Webcam' or 'Upload Video File' in the sidebar to process your first session!")
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
