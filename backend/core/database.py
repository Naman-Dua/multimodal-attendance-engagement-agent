from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///./classroom_sessions.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(String, index=True)
    session_name = Column(String)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    average_attention = Column(Float, default=0.0)
    attendance_count = Column(Integer, default=0)

    persons = relationship("SessionPerson", back_populates="session")
    metrics = relationship("PersonMetric", back_populates="session")

class SessionPerson(Base):
    __tablename__ = "session_persons"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True)
    person_id = Column(String, index=True)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    presence_duration = Column(Float, default=0.0)  # in seconds
    is_present = Column(Boolean, default=False)

    session = relationship("Session", back_populates="persons")

class PersonMetric(Base):
    __tablename__ = "person_metrics"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True)
    person_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    attention = Column(Float, default=0.0)
    emotion = Column(String, default="neutral")

    session = relationship("Session", back_populates="metrics")

Base.metadata.create_all(bind=engine)
