# 🤖 AI-Powered Resume Matcher and Job Recommender

## 📌 Project Overview  
The **AI-Powered Resume Matcher and Job Recommender** is a smart job discovery platform that leverages Natural Language Processing (NLP) to analyze resumes and recommend suitable job opportunities. Instead of manually browsing multiple job portals, users can upload their resume and instantly receive job suggestions based on their skills, education, and experience — without the need for profile creation or direct submission.

---

## 🎯 Key Features
- 📄 Upload resume in **PDF** or **DOCX** format  
- 🧠 Extract skills, education, and experience using **NLP**  
- 🔍 Get real-time job suggestions via **JSearch API**  
- 💡 Personalized recommendations based on resume content  
- 🔐 Secure login and password handling using **Werkzeug & bcrypt**  
- 📬 Direct redirection to official job application sites  
- 🖥️ Simple and user-friendly interface for seamless navigation  

---

## 🧭 Project Workflow

1. 🏠 **Landing Page**:  
   - Contains a job search bar that allows **any visitor** to search for jobs without logging in.  
   - Displays “Sign Up” and “Login” buttons for account access.

2. 📝 **Sign Up**:  
   - New users provide **academic and personal details** (e.g., name, college, branch, year, etc.).  
   - After successful registration, users are redirected to the login page.

3. 🔐 **Login**:  
   - Users log in using their registered email and password.  
   - On success, they are redirected to the **user dashboard**.

4. 📊 **Dashboard**:  
   - Provides a search bar for **manual job searches** by title, location, or category.  
   - Offers an **"Upload Resume"** button for personalized job recommendations.

5. 📄 **Resume Upload and Parsing**:  
   - The uploaded resume is parsed using NLP to extract skills, experience, and education.  
   - Based on this information, **AI-powered job suggestions** are displayed directly on the dashboard.

6. 🔎 **Job Suggestions**:  
   - Jobs are fetched in real-time using the **JSearch API**.  
   - Clicking "Apply Now" redirects users to the official job listing for application.

---

## 🛠️ Tech Stack

| Area          | Tools & Libraries                            |
|---------------|-----------------------------------------------|
| **Frontend**  | HTML, CSS, JavaScript                         |
| **Backend**   | Python, Flask, Jinja2                         |
| **Database**  | SQLite, SQLAlchemy ORM                        |
| **NLP**       | spaCy, NLTK, PyPDF2, pdfminer.six, pytesseract|
| **Security**  | bcrypt, cryptography, python-dotenv           |
| **Job Search**| JSearch API (via RapidAPI)                    |

---
