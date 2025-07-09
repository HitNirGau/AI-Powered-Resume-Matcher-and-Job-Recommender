from app import db
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from app import app
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    passhash = db.Column(db.String(256), nullable=False)
    dob = db.Column(db.Date, nullable=True)

    resume = db.Column(db.LargeBinary, nullable=True)  # Storing resume as binary
    resume_filename = db.Column(db.String(255), nullable=True)

    university = db.Column(db.String(255), nullable=True)  # Optional university field
    study_year = db.Column(db.String(50), nullable=True)  # Optional study year field
    work_experience = db.Column(db.Text, nullable=True)

    extracted_keywords = db.Column(db.String(500), nullable=True)
    jobs = db.relationship('JobRecommendation', backref=db.backref('user', lazy=True))

class JobRecommendation(db.Model):
    __tablename__ = 'job_recommendations'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    job_title = db.Column(db.String(200), nullable=False)
    company = db.Column(db.String(200), nullable=False)
    location = db.Column(db.String(200))
    description = db.Column(db.Text)
    apply_link = db.Column(db.String(500))  # Add this field for application links
    keywords = db.Column(db.Text)  # Add this field to store keywords used
    job_id = db.Column(db.String(100))  # Add this field to store unique job IDs
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


    def __repr__(self):
        return f'<JobRecommendation {self.job_title} at {self.company}>'

with app.app_context():
    db.create_all()
