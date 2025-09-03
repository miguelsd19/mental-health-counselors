from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .db import Base


class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    associations = relationship("QAAssociation", back_populates="question")


class Answer(Base):
    __tablename__ = "answers"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    associations = relationship("QAAssociation", back_populates="answer")


class QAAssociation(Base):
    __tablename__ = "qa_associations"
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    answer_id = Column(Integer, ForeignKey("answers.id"), nullable=False)
    source = Column(String(255), nullable=True)
    tags = Column(String(255), nullable=True) 

    question = relationship("Question", back_populates="associations")
    answer = relationship("Answer", back_populates="associations")