import streamlit as st
import pandas as pd
from datetime import datetime
import json

from backend.core.database import SessionLocal, Session as SessionModel, SessionPerson, PersonMetric
from backend.core.analytics_service import AnalyticsService
from backend.core.teacher_profiles import TeacherProfileService
from backend.core.ai_suggestions import AISuggestionEngine
from backend.core.recommendations_engine import RecommendationsEngine
from backend.pipeline import ClassroomPipeline

# Page configuration
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

# Title and header
st.title("🎓 Multimodal Attendance & Engagement Dashboard")
st.caption("AI-Powered Privacy-Preserving Classroom Intelligence System")

# Sidebar - Session Navigation & Controls
st.sidebar.header("Classroom Session Control")

db = SessionLocal()
sessions = []
try:
    sessions = db.query(SessionModel).order_by(SessionModel.id.desc()).all()
finally:
    db.close()

session_options = {f"Session #{s.id} - {s.session_name} ({s.start_time.strftime('%Y-%m-%d %H:%M')})": s.id for s in sessions}

if session_options:
    selected_label = st.sidebar.selectbox("Select Classroom Session:", list(session_options.keys()))
    selected_session_id = session_options[selected_label]
else:
    st.sidebar.info("No recorded sessions found in database.")
    selected_session_id = None

st.sidebar.markdown("---")
st.sidebar.subheader("Live Simulation")
if st.sidebar.button("▶ Run Demo Simulation (30 Frames)"):
    with st.spinner("Processing multimodal feed..."):
        pipeline = ClassroomPipeline(session_name="Demo Live Lecture", teacher_id="PROF_SHARMA")
        pipeline.run(max_frames=30, synthetic_test=True)
    st.sidebar.success("Simulation complete! Refreshing...")
    st.rerun()

# Main Dashboard Content
if selected_session_id:
    db = SessionLocal()
    session_data = None
    persons = []
    try:
        session_data = db.query(SessionModel).filter(SessionModel.id == selected_session_id).first()
        persons = db.query(SessionPerson).filter(SessionPerson.session_id == selected_session_id).all()
    finally:
        db.close()

    # Top Metric KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Confirmed Attendance", f"{session_data.attendance_count if session_data else 0} Students")
    with col2:
        st.metric("Avg Attention Index", f"{session_data.average_attention if session_data else 0.0:.1f}%")
    with col3:
        st.metric("Teacher ID", session_data.teacher_id if session_data else "N/A")
    with col4:
        st.metric("Session Status", "Completed" if session_data and session_data.end_time else "Active")

    st.markdown("---")

    # Tabs for Organized Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Attendance Records", 
        "📈 Engagement Timeline", 
        "💡 Instructor Suggestions", 
        "👤 Student Analytics", 
        "👩‍🏫 Teacher Profile"
    ])

    # Tab 1: Attendance Records
    with tab1:
        st.subheader("Automated Presence-Based Attendance")
        st.caption("Attendance marked based on sustained presence duration rather than single-frame detection.")
        if persons:
            records = []
            for p in persons:
                records.append({
                    "Participant ID": p.person_id,
                    "First Detected": p.first_seen.strftime("%H:%M:%S") if p.first_seen else "N/A",
                    "Last Detected": p.last_seen.strftime("%H:%M:%S") if p.last_seen else "N/A",
                    "Cumulative Duration (s)": round(p.presence_duration, 1),
                    "Attendance Status": "✅ Confirmed Present" if p.is_present else "❌ Incomplete Presence"
                })
            df_att = pd.DataFrame(records)
            st.dataframe(df_att, width="stretch")
        else:
            st.info("No participant records registered for this session.")

    # Tab 2: Engagement Timeline
    with tab2:
        st.subheader("Temporal Engagement Dynamics")
        st.caption("Multimodal fusion of visual attention and audio interaction over time.")
        trends = analytics_service.get_session_trends(selected_session_id)
        timeline = trends.get("timeline", [])

        if timeline:
            df_timeline = pd.DataFrame(timeline)
            df_chart = df_timeline.set_index("timestamp")[["attention"]]
            st.line_chart(df_chart, width="stretch")
            
            with st.expander("Explore Raw Temporal Points"):
                st.dataframe(df_timeline, width="stretch")
        else:
            st.info("No granular engagement telemetry points recorded for this session.")

    # Tab 3: Instructor Suggestions
    with tab3:
        st.subheader("Context-Aware AI Teaching Suggestions")
        st.caption("Decision-support recommendations generated from multimodal classroom signals.")
        
        # Real-time warnings
        mock_status = {
            "people": [{"id": p.person_id, "attention": int(session_data.average_attention if session_data else 70)} for p in persons]
        }
        recs = recs_engine.generate_realtime_recommendations(mock_status)
        if recs:
            st.markdown("#### Real-Time Alerts")
            for r in recs:
                if r['type'] == 'alert':
                    st.error(f"🚨 {r['message']}")
                elif r['type'] == 'warning':
                    st.warning(f"⚠️ {r['message']}")
                else:
                    st.info(f"ℹ️ {r['message']}")

        # Session-level AI suggestions
        st.markdown("#### AI Pedagogical Guidance (LLM Engine)")
        suggestions_data = ai_engine.generate_suggestions({
            "attendance_count": session_data.attendance_count if session_data else 0,
            "average_engagement": (session_data.average_attention / 100.0) if session_data else 0.7,
            "audio_level": "moderate"
        })
        
        for tip in suggestions_data.get("suggestions", []):
            st.markdown(f"- 🎯 **{tip}**")

    # Tab 4: Student Analytics
    with tab4:
        st.subheader("Reflective Student Session Analytics")
        st.caption("Privacy-preserving individual focus summary without competitive ranking or grading.")
        
        if persons:
            student_ids = [p.person_id for p in persons]
            chosen_student = st.selectbox("Select Student:", student_ids)
            student_report = analytics_service.get_student_session_analytics(selected_session_id, chosen_student)
            
            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                st.metric("Attendance Confirmed", "Yes" if student_report.get("attendance_confirmed") else "No")
            with scol2:
                st.metric("Total Duration", f"{student_report.get('presence_duration_seconds', 0)} sec")
            with scol3:
                st.metric("Focus Consistency", f"{student_report.get('focus_consistency_percentage', 0)}%")

            if student_report.get("timeline"):
                st.markdown("##### Personal Attention Timeline")
                df_student = pd.DataFrame(student_report["timeline"]).set_index("time")
                st.line_chart(df_student, width="stretch")
        else:
            st.info("No student telemetry available.")

    # Tab 5: Teacher Profile
    with tab5:
        st.subheader("Teacher Effectiveness & Analytics Profile")
        teacher_id = session_data.teacher_id if session_data else "TEACHER_01"
        profile = teacher_service.get_profile(teacher_id)
        
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.metric("Instructional Effectiveness Score", f"{profile['effectiveness_score']}%", f"+{profile['trend']}% Trend")
            st.markdown("##### Strengths Identified")
            for s in profile.get("strengths", []):
                st.markdown(f"- ⭐ {s}")
        with tcol2:
            st.markdown("##### Recent Class Engagement History")
            st.dataframe(pd.DataFrame(profile.get("class_history", [])), width="stretch")

        st.markdown("##### Longitudinal Student Progress Trends")
        st.dataframe(pd.DataFrame(profile.get("student_trends", [])), width="stretch")

else:
    st.info("Click 'Run Demo Simulation' in the left sidebar to generate your first session dataset!")
