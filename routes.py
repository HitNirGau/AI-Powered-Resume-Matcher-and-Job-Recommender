from flask import render_template, url_for, redirect, flash, request, session, Blueprint
from datetime import datetime
from models import db, User, JobRecommendation
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from config import Config
import requests
import io
import os
import re
import json

# Document Parsing
import pdfplumber
import PyPDF2
import docx2txt
from docx import Document
from PIL import Image
import pytesseract
import docx
# Natural Language Processing
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.probability import FreqDist
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import string

routes = Blueprint('routes', __name__)

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Expanded irrelevant terms to filter out
IRRELEVANT_TERMS = {
    'student', 'foundation', 'member', 'cgpa', 'branch', 'branchb', 'knowledge', 'basics', 'experience',
    'skills', 'projects', 'computer', 'engineering', 'tourism', 'background', 'proficient', 'familiar',
    'exposure', 'introduction', 'basic', 'advanced', 'intermediate', 'beginner', 'expert', 'proficiency',
    'understanding', 'working', 'education', 'university', 'college', 'school', 'institute', 'degree',
    'bachelor', 'master', 'phd', 'interests', 'hobby', 'contact', 'email', 'phone', 'address', 'resume',
    'curriculum', 'vitae', 'cv', 'position', 'job', 'career', 'objective', 'summary', 'profile'
}

# Tech skills with proper casing (important for technologies like JavaScript)
TECH_SKILLS_WITH_CASING = {
    # Programming Languages
    'JavaScript': ['javascript', 'js'],
    'Python': ['python', 'py'],
    'Java': ['java'],
    'C++': ['c++', 'cpp', 'c plus plus'],
    'C#': ['c#', 'csharp', 'c sharp'],
    'PHP': ['php'],
    'Ruby': ['ruby', 'rb'],
    'Swift': ['swift'],
    'Kotlin': ['kotlin'],
    'Go': ['go', 'golang'],
    'Rust': ['rust'],
    'TypeScript': ['typescript', 'ts'],
    'Scala': ['scala'],
    'Perl': ['perl'],
    'R': ['r programming', 'r language'],
    'MATLAB': ['matlab'],
    'Bash': ['bash', 'shell scripting'],
    'PowerShell': ['powershell'],

    # Web Development
    'HTML': ['html', 'html5'],
    'CSS': ['css', 'css3'],
    'SASS': ['sass', 'scss'],
    'LESS': ['less'],
    'Bootstrap': ['bootstrap'],
    'Tailwind CSS': ['tailwind', 'tailwindcss', 'tailwind css'],
    'jQuery': ['jquery'],
    'Angular': ['angular', 'angularjs', 'angular.js'],
    'Vue.js': ['vue', 'vuejs', 'vue.js'],
    'React': ['react', 'reactjs', 'react.js'],
    'Redux': ['redux'],
    'Webpack': ['webpack'],
    'Babel': ['babel'],
    'Vite': ['vite'],

    # Backend & Frameworks
    'Node.js': ['node', 'nodejs', 'node.js'],
    'Express.js': ['express', 'expressjs', 'express.js'],
    'Flask': ['flask'],
    'Django': ['django'],
    'Spring': ['spring', 'spring boot', 'springboot'],
    'Laravel': ['laravel'],
    'Ruby on Rails': ['rails', 'ruby on rails', 'ror'],
    'FastAPI': ['fastapi'],
    'ASP.NET': ['asp.net', 'aspnet'],
    'Next.js': ['next', 'nextjs', 'next.js'],
    'Jinja2': ['jinja', 'jinja2'],
    'RESTful API': ['rest', 'restful', 'rest api', 'restful api'],

    # Databases
    'SQL': ['sql'],
    'MySQL': ['mysql'],
    'PostgreSQL': ['postgresql', 'postgres'],
    'MongoDB': ['mongodb', 'mongo'],
    'SQLite': ['sqlite'],
    'Oracle': ['oracle', 'oracle db'],
    'Redis': ['redis'],
    'Firebase': ['firebase'],

    # DevOps & Cloud
    'AWS': ['aws', 'amazon web services'],
    'Azure': ['azure', 'microsoft azure'],
    'GCP': ['gcp', 'google cloud', 'google cloud platform'],
    'Docker': ['docker'],
    'Kubernetes': ['kubernetes', 'k8s'],
    'Jenkins': ['jenkins'],
    'Terraform': ['terraform'],
    'Git': ['git'],
    'GitHub': ['github'],
    'GitLab': ['gitlab'],
    'Bitbucket': ['bitbucket'],

    # Data Science/AI
    'Machine Learning': ['machine learning', 'ml'],
    'Deep Learning': ['deep learning', 'dl'],
    'TensorFlow': ['tensorflow', 'tf'],
    'PyTorch': ['pytorch'],
    'Keras': ['keras'],
    'Scikit-learn': ['scikit-learn', 'sklearn'],
    'Pandas': ['pandas'],
    'NumPy': ['numpy'],
    'Data Science': ['data science'],
    'Data Analysis': ['data analysis', 'data analytics'],
    'Data Visualization': ['data visualization', 'data viz'],
    'NLP': ['nlp', 'natural language processing'],
    'Computer Vision': ['computer vision', 'cv'],

    # Testing
    'Jest': ['jest'],
    'Mocha': ['mocha'],
    'Selenium': ['selenium'],
    'Cypress': ['cypress'],
    'PyTest': ['pytest'],
    'TDD': ['tdd', 'test driven development'],

    # Project Management
    'Agile': ['agile', 'agile methodology'],
    'Scrum': ['scrum'],
    'Kanban': ['kanban'],
    'Jira': ['jira'],
    'Confluence': ['confluence'],

    # Common Frameworks/Stacks
    'MERN Stack': ['mern', 'mern stack'],
    'MEAN Stack': ['mean', 'mean stack'],
    'LAMP Stack': ['lamp', 'lamp stack'],

    # Domain specific terms
    'Microservices': ['microservices', 'microservice architecture'],
    'CI/CD': ['ci/cd', 'continuous integration', 'continuous deployment'],
    'REST API': ['rest api', 'restful api', 'rest'],
    'GraphQL': ['graphql'],
    'ORM': ['orm', 'object relational mapping'],
    'Disease Management': ['disease management', 'healthcare it'],
    'Tourism': ['tourism management', 'travel tech'],
}

# Create a reverse mapping for lookup
TECH_SKILLS_LOOKUP = {}
for standard_term, variations in TECH_SKILLS_WITH_CASING.items():
    for variation in variations:
        TECH_SKILLS_LOOKUP[variation] = standard_term

# Create a set of all variations for quick membership testing
ALL_TECH_VARIATIONS = set()
for variations in TECH_SKILLS_WITH_CASING.values():
    ALL_TECH_VARIATIONS.update(variations)


def extract_text_from_pdf(pdf_bytes):
    text = ""
    try:
        # Ensure pdf_bytes is treated correctly
        if isinstance(pdf_bytes, io.BytesIO):
            pdf_bytes.seek(0)  # Reset cursor position
        else:
            pdf_bytes = io.BytesIO(pdf_bytes)  # Convert raw bytes to BytesIO

        pdf_reader = PyPDF2.PdfReader(pdf_bytes)

        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")

    return text.strip()


def extract_text_from_docx(docx_bytes):
    text = ""
    try:
        if isinstance(docx_bytes, bytes):
            docx_bytes = io.BytesIO(docx_bytes)

        doc = docx.Document(docx_bytes)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
    return text.strip()


def clean_and_normalize_text(text):
    """Clean and normalize text for better keyword extraction"""
    # Convert to lowercase for matching
    text = text.lower()

    # Fix common issues with technical terms
    text = text.replace('c ++', 'c++')
    text = text.replace('c #', 'c#')
    text = text.replace('java script', 'javascript')
    text = text.replace('type script', 'typescript')

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Replace newlines with spaces
    text = text.replace('\n', ' ')

    # Remove punctuation that's not part of technical terms
    text = re.sub(r'[^\w\s\+\#\.\-]', ' ', text)

    return text


def extract_ngrams(text, min_n=1, max_n=3):
    """Extract n-grams from text to capture multi-word phrases"""
    words = text.split()
    ngrams = []

    # Generate n-grams of different lengths
    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i + n])
            ngrams.append(ngram)

    return ngrams


def normalize_keyword(keyword):
    """Normalize a keyword to its standard form if known"""
    return TECH_SKILLS_LOOKUP.get(keyword.lower(), keyword)


def find_technical_terms(text):
    """Find technical terms in the text using pattern matching"""
    cleaned_text = clean_and_normalize_text(text)
    ngrams = extract_ngrams(cleaned_text, min_n=1, max_n=3)

    # Find matches in our dictionary
    matches = {}
    for ngram in ngrams:
        # Skip very short terms unless they're exact matches (like 'R')
        if len(ngram) < 2 and ngram.lower() not in ALL_TECH_VARIATIONS:
            continue

        if ngram.lower() in ALL_TECH_VARIATIONS:
            standard_term = normalize_keyword(ngram)
            # Count frequency
            matches[standard_term] = matches.get(standard_term, 0) + 1

    return matches


def extract_keywords_from_sections(text, num_keywords=25):
    """Extract keywords specifically from resume sections that are likely to contain skills"""
    # Define common section headers in resumes
    section_patterns = [
        r'(?i)skills|technologies|technical skills|programming languages|tools|frameworks',
        r'(?i)experience|work experience|employment|professional experience',
        r'(?i)projects|technical projects|project experience',
        r'(?i)education|academic|qualifications'
    ]

    # Split the text into sections
    sections = []
    current_section = ""

    # Try to parse the document structure
    lines = text.split('\n')
    for line in lines:
        # Check if this line could be a section header
        is_header = False
        for pattern in section_patterns:
            if re.search(pattern, line, re.IGNORECASE) and len(line) < 50:  # Headers are usually short
                if current_section:
                    sections.append(current_section)
                current_section = line + "\n"
                is_header = True
                break

        if not is_header and current_section:
            current_section += line + "\n"

    # Add the last section
    if current_section:
        sections.append(current_section)

    # If no clear sections were found, treat the whole text as one section
    if not sections:
        sections = [text]

    # Process each section, giving more weight to skills sections
    all_keywords = {}

    for section in sections:
        # Check if this looks like a skills section
        is_skills_section = any(re.search(pattern, section[:50], re.IGNORECASE)
                                for pattern in [r'(?i)skills', r'(?i)technologies'])

        # Extract technical terms from this section
        section_keywords = find_technical_terms(section)

        # Give more weight to terms found in skills sections
        for keyword, count in section_keywords.items():
            weight = 3 if is_skills_section else 1
            all_keywords[keyword] = all_keywords.get(keyword, 0) + (count * weight)

    # Sort by frequency
    sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)

    # Filter out non-technical/irrelevant terms
    filtered_keywords = [(kw, count) for kw, count in sorted_keywords
                         if kw.lower() not in IRRELEVANT_TERMS]

    # Return only the keywords (not the counts)
    return [kw for kw, _ in filtered_keywords[:num_keywords]]


def extract_keywords(text, num_keywords=25):
    """
    Main function to extract keywords from resume text
    """
    # First try section-based extraction
    section_keywords = extract_keywords_from_sections(text, num_keywords)

    # If we got enough keywords, use those
    if len(section_keywords) >= 10:
        return section_keywords

    # Otherwise fall back to the whole document analysis
    clean_text = clean_and_normalize_text(text)

    # Extract tech terms from the whole document
    tech_terms = find_technical_terms(clean_text)

    # Sort by frequency
    sorted_terms = sorted(tech_terms.items(), key=lambda x: x[1], reverse=True)

    # Take top keywords
    keywords = [term for term, _ in sorted_terms[:num_keywords]]

    # If still not enough, use TF-IDF as a last resort
    if len(keywords) < 10:
        try:
            # Tokenize text
            tokens = word_tokenize(clean_text)
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [w for w in tokens if w.lower() not in stop_words and len(w) > 2]

            # Create sentences for TF-IDF
            sentences = sent_tokenize(text)
            if sentences:
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()

                # Get important words based on TF-IDF scores
                tfidf_scores = {}
                for i, sentence in enumerate(sentences):
                    for j, word in enumerate(feature_names):
                        if tfidf_matrix[i, j] > 0:
                            tfidf_scores[word] = tfidf_scores.get(word, 0) + tfidf_matrix[i, j]

                # Sort by TF-IDF score
                sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)

                # Add high TF-IDF words that aren't in keywords yet
                for word, _ in sorted_words:
                    if word.lower() not in IRRELEVANT_TERMS and word not in keywords:
                        keywords.append(word)
                        if len(keywords) >= num_keywords:
                            break
        except Exception as e:
            print(f"TF-IDF extraction failed: {e}")

    return keywords[:num_keywords]


def parse_resume(resume_bytes, filename):
    """
    Parses the resume file, extracts text, and returns keywords.
    :param resume_bytes: Resume file in bytes
    :param filename: Name of the uploaded file
    :return: Extracted keywords as a comma-separated string
    """
    text = ""

    try:
        if filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(resume_bytes)
        elif filename.lower().endswith((".docx", ".doc")):
            text = extract_text_from_docx(resume_bytes)
        else:
            print(f"Unsupported file format: {filename}")
    except Exception as e:
        print(f"Error parsing file {filename}: {e}")

    if text:
        try:
            keywords = extract_keywords(text, num_keywords=30)  # Extract more keywords for better matching
            # Use proper casing for keywords
            keywords = [k if k in TECH_SKILLS_WITH_CASING else k for k in keywords]
            return ", ".join(keywords)  # Store as comma-separated string
        except Exception as e:
            print(f"Error extracting keywords: {e}")

    return ""  # Return empty if no keywords found


def update_missing_keywords():
    users = User.query.filter((User.extracted_keywords == None) | (User.extracted_keywords == "")).all()

    for user in users:
        if user.resume and user.resume_filename:
            try:
                resume_bytes = io.BytesIO(user.resume)
                extracted_keywords = parse_resume(resume_bytes, user.resume_filename)

                if extracted_keywords:  # Only update if we successfully extract keywords
                    user.extracted_keywords = extracted_keywords
                    db.session.commit()
                    print(f"Updated keywords for {user.email}")
            except Exception as e:
                print(f"Error processing {user.email}: {e}")


#-------------------------------#

# Decorator for auth_required
def auth_required(func):
    @wraps(func)
    def inner(*args, **kwargs):
        if 'user_id' in session:
            return func(*args, **kwargs)
        else:
            flash('Please log in to continue', "warning")
            return redirect(url_for('routes.login'))
    return inner


@routes.route('/')
def home():
    return render_template('home.html')


@routes.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        university = request.form.get("university", "").strip()
        study_year = request.form.get("study_year", "").strip()
        dob = request.form.get("dob")
        resume_file = request.files.get("resume")

        if not all([full_name, email, password, university, study_year]):
            flash("Please fill in all required fields.", "danger")
            return redirect(url_for("routes.register"))

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Invalid email format.", "danger")
            return redirect(url_for("routes.register"))

        try:
            dob_obj = datetime.strptime(dob, "%Y-%m-%d").date() if dob else None
        except ValueError:
            flash("Invalid date format. Please enter a valid date.", "danger")
            return redirect(url_for("routes.register"))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered. Try logging in.", "danger")
            return redirect(url_for("routes.register"))

        hashed_password = generate_password_hash(password)

        resume_data = None
        resume_filename = None
        extracted_keywords = ""

        if resume_file and resume_file.filename:
            try:
                resume_data = resume_file.read()
                resume_filename = resume_file.filename.lower()
                extracted_keywords = parse_resume(io.BytesIO(resume_data), resume_filename)
            except Exception as e:
                flash(f"Error processing resume: {str(e)}", "danger")
                return redirect(url_for("routes.register"))

        new_user = User(
            full_name=full_name,
            email=email,
            passhash=hashed_password,
            dob=dob_obj,
            university=university,
            study_year=study_year,
            resume=resume_data,
            resume_filename=resume_filename,
            extracted_keywords=extracted_keywords  # Store extracted keywords
        )

        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for("routes.login"))
        except Exception as e:
            db.session.rollback()
            flash("An error occurred during registration. Please try again.", "danger")
            print("Error:", e)

    return render_template("register.html")


@routes.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        # Validate input
        if not email or not password:
            flash('Please enter both email and password.', 'danger')
            return render_template('login.html')

        # Check if user exists
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.passhash, password):
            # Store user session
            session['user_id'] = user.id
            session['user_name'] = user.full_name  # Optional: Store name for personalized greeting

            flash('Login successful!', 'success')
            return redirect(url_for('routes.dashboard'))  # Redirect to dashboard/homepage

        else:
            flash('Invalid email or password. Please try again.', 'danger')

    return render_template('login.html')

@routes.route("/search-jobs", methods=["GET"])
def search_jobs():
    try:
        # Validate and sanitize input
        job_title = request.args.get("query", "").strip() or "developer"
        location = request.args.get("area", "").strip() or "chicago"
        category = request.args.get("category", "").strip()

        # Validate input lengths
        if len(job_title) > 100 or len(location) > 100:
            flash("Invalid search parameters", "danger")
            return redirect(url_for('routes.home'))

        url = "https://jsearch.p.rapidapi.com/search"

        params = {
            "query": f"{job_title} jobs in {location}",
            "page": "1",
            "num_pages": "1",
            "country": "in",
            "date_posted": "all"
        }

        # If a category is selected, add it to the query
        if category:
            params["employment_types"] = category.upper()  # API expects uppercase

        headers = {
            "X-RapidAPI-Key": Config.RAPIDAPI_KEY,
            "X-RapidAPI-Host": Config.RAPIDAPI_HOST
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
        except requests.RequestException as e:
            flash(f"Network error: {str(e)}", "danger")
            return redirect(url_for('routes.home'))

        try:
            job_data = response.json()
            jobs = job_data.get("data", [])

            # Process jobs to ensure company name is not empty
            for job in jobs:
                # Try multiple keys for company name
                company_keys = ['company_name', 'employer_name', 'company']
                for key in company_keys:
                    if job.get(key):
                        job['display_company'] = job[key]
                        break
                else:
                    # If no company name found, use a more descriptive default
                    job['display_company'] = f"{job_title} Employer"

            # Additional error handling for no jobs
            if not jobs:
                flash("No jobs found for the given search criteria.", "warning")
                return redirect(url_for('routes.home'))

        except ValueError:
            flash("Failed to parse job data", "danger")
            return redirect(url_for('routes.home'))

        return render_template("home.html",
                               jobs=jobs,
                               query=job_title,
                               area=location,
                               category=category)

    except Exception as e:
        # Catch-all for any unexpected errors
        flash(f"An unexpected error occurred: {str(e)}", "danger")
        return redirect(url_for('routes.home'))

@routes.route('/all_jobs/<int:user_id>')
@auth_required
def all_jobs(user_id):
    user = User.query.get(user_id)
    return render_template('search.html', user=user)

@routes.route('/logout')
@auth_required
def logout():
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('routes.login'))

@routes.route('/dashboard')
@auth_required
def dashboard():
    user = User.query.get(session['user_id'])
    recommendations = JobRecommendation.query.filter_by(user_id=user.id).order_by(JobRecommendation.created_at.desc()).all()
    return render_template('dashboard.html',
                            user = user,
                           recommendations = recommendations,
                            parse_error = False
                           )

@routes.route('/profile/<int:user_id>')
@auth_required
def profile(user_id):
    user = User.query.get(user_id)
    return render_template('profile.html', user=user)


from flask import request, redirect, url_for, flash
from flask_login import current_user
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import io


@routes.route('/update_profile/<int:user_id>', methods=['POST'])
@auth_required
def update_profile(user_id):
    user = User.query.get(user_id)

    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("routes.profile", user_id=user_id))

    full_name = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    update_password = request.form.get("update_password", "").strip()
    university = request.form.get("university", "").strip()
    study_year = request.form.get("study_year", "").strip()
    work_experience = request.form.get("work_experience", "").strip()
    dob = request.form.get("dob")
    resume_file = request.files.get("resume")

    try:
        dob_obj = datetime.strptime(dob, "%Y-%m-%d").date() if dob else None
    except ValueError:
        flash("Invalid date format. Please enter a valid date.", "danger")
        return redirect(url_for("routes.profile", user_id=user_id))

    if not check_password_hash(user.passhash, password):
        flash("Incorrect password. Please try again.", "danger")
        return redirect(url_for("routes.profile", user_id=user_id))

    if update_password:
        updated_hashed_password = generate_password_hash(update_password)
        user.passhash = updated_hashed_password

    user.full_name = full_name
    user.email = email
    user.work_experience = work_experience
    user.dob = dob_obj
    user.university = university
    user.study_year = study_year

    extracted_keywords = None

    # Process resume file if uploaded
    if resume_file and resume_file.filename:
        try:
            file_ext = resume_file.filename.split(".")[-1].lower()
            resume_bytes = io.BytesIO(resume_file.read())  # Convert file to bytes

            extracted_keywords = parse_resume(resume_bytes, resume_file.filename)
            print(f"Extracted keywords: {extracted_keywords}")

            if extracted_keywords:
                user.resume = resume_bytes.getvalue()  # Store binary data
                user.resume_filename = resume_file.filename
                user.extracted_keywords = extracted_keywords  # Store extracted keywords
            else:
                print("No keywords were extracted from the resume")

        except Exception as e:
            print(f"Resume processing error: {str(e)}")
            flash(f"Error processing resume: {str(e)}", "danger")

    # Use existing keywords if no new resume was uploaded or if extraction failed
    if not extracted_keywords and user.extracted_keywords:
        extracted_keywords = user.extracted_keywords
        print(f"Using existing keywords: {extracted_keywords}")

    # Process job recommendations if we have keywords (either from new upload or existing)
    if extracted_keywords:
        try:
            # Remove old job recommendations
            JobRecommendation.query.filter_by(user_id=user.id).delete()

            # Fetch new job recommendations
            recommended_jobs = fetch_job_recommendations(extracted_keywords)
            print(f"Recommended jobs count: {len(recommended_jobs)}")

            # Store new job recommendations
            for job in recommended_jobs:
                new_job = JobRecommendation(
                    user_id=user.id,
                    job_title=job.get('title', 'N/A'),
                    company=job.get('company', 'N/A'),
                    location=job.get('location', 'N/A'),
                    description=job.get('description', '')[:500],# Limit to 500 chars
                    apply_link=job.get('apply_link') or (
                                    job.get('job_google_link') or
                                    (job.get('job_publisher', [{}])[0].get('apply_link') if isinstance(
                                        job.get('job_publisher'), list) else 'No link available')
                            ),
                    job_id=job.get('job_id'),
                    keywords=extracted_keywords
                )
                db.session.add(new_job)

            print(f"Added {len(recommended_jobs)} job recommendations to session")
        except Exception as e:
            print(f"Job recommendation error: {str(e)}")
            flash(f"Error processing job recommendations: {str(e)}", "danger")

    # Commit all changes to the database
    try:
        db.session.commit()
        print("Database changes committed successfully")
        flash("Profile updated successfully!", "success")
    except Exception as e:
        db.session.rollback()
        print(f"Database commit error: {str(e)}")
        flash(f"Error saving changes: {str(e)}", "danger")

    return redirect(url_for("routes.dashboard", user_id=user_id))

@routes.route('/upload_resume', methods=['POST'])
@auth_required
def upload_resume():
    if 'user_id' not in session:
        return redirect(url_for('routes.login'))

    user = User.query.get(session['user_id'])
    resume_file = request.files.get("resume")

    if not resume_file or not resume_file.filename:
        flash("No file selected. Please upload a resume.", "warning")
        return redirect(url_for("routes.dashboard"))

    try:
        resume_data = resume_file.read()
        resume_filename = resume_file.filename.lower()

        extracted_text = parse_resume(resume_data, resume_filename)
        extracted_keywords = extract_keywords(extracted_text)

        user.resume = resume_data
        user.resume_filename = resume_filename
        user.extracted_keywords = ", ".join(extracted_keywords)  # Store keywords as a comma-separated string

        db.session.commit()

        keywords_str = keywords_str = ", ".join(extracted_keywords)

        if keywords_str:
            # Store keywords in session for reference
            session['resume_keywords'] = keywords_str

            # Fetch job recommendations based on keywords
            jobs = fetch_job_recommendations(keywords_str)

            if jobs:
                # Clear previous recommendations
                JobRecommendation.query.filter_by(user_id=user.id).delete()

                # Save new recommendations to database
                for job in jobs:
                    new_recommendation = JobRecommendation(
                        user_id=user.id,
                        job_title=job['title'],
                        company=job['company'],
                        location=job['location'],
                        description=job['description'][:500], # Limit description length
                        apply_link = job.get('apply_link') or (
                                    job.get('job_google_link') or
                                    (job.get('job_publisher', [{}])[0].get('apply_link') if isinstance(
                                        job.get('job_publisher'), list) else 'No link available')
                            ),
                        job_id = job.get('job_id'),
                        keywords = keywords_str
                    )
                    print("Job Apply Link:", job.get('apply_link'))
                    print("Job ID:", job.get('job_id'))
                    print("Keywords used:", extracted_keywords)
                    db.session.add(new_recommendation)

                db.session.commit()
                flash('Resume uploaded successfully! Here are your job recommendations.', 'success')
            else:
                flash('Resume uploaded, but no job recommendations found. Try refining your skills.', 'warning')
        else:
            session['parse_error'] = True
            flash('Unable to parse your resume. Please make sure it\'s in a supported format (PDF, DOCX).', 'danger')

    except Exception as e:
        flash(f'Error processing resume: {str(e)}', 'danger')

    return redirect(url_for('routes.dashboard'))


def fetch_job_recommendations(keywords_str, num_jobs=5):
    """
    Fetch job recommendations from JSSearch API based on extracted keywords.
    Tries multiple keywords until results are found.
    """
    import json

    print("DEBUG: RAPIDAPI_KEY =", Config.RAPIDAPI_KEY)
    print("DEBUG: RAPIDAPI_HOST =", Config.RAPIDAPI_HOST)

    try:
        api_key = Config.RAPIDAPI_KEY
        host = Config.RAPIDAPI_HOST

        if not api_key:
            print("JSSearch API key is missing")
            return []

        if not host:
            raise ValueError("Missing RAPIDAPI_HOST in config")

        keywords = keywords_str.split(', ')
        technical_keywords = [k for k in keywords if k.lower() in ALL_TECH_VARIATIONS]
        if not technical_keywords:
            technical_keywords = keywords  # fallback

        base_url = "https://" + host if not host.startswith("http") else host
        url = base_url.rstrip("/") + "/search"

        headers = {
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }

        # Try each keyword until we get results
        for query in technical_keywords:
            params = {
                "query": query,
                "page": "1",
                "num_pages": "1"
            }

            print("Trying query:", query)
            print("Requesting URL:", url)
            print("Params:", params)

            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                print("Response JSON:", json.dumps(data, indent=2))

                if 'data' in data and data['data']:
                    jobs = []
                    for job in data['data'][:num_jobs]:
                        job_info = {
                            'title': job.get('job_title', 'No title'),
                            'company': job.get('employer_name', 'Unknown company'),
                            'location': job.get('job_city', '') + ', ' + job.get('job_country', ''),
                            'description': job.get('job_description', 'No description available'),
                            'apply_link': job.get('apply_link') or (
                                    job.get('job_google_link') or
                                    (job.get('job_publisher', [{}])[0].get('apply_link') if isinstance(
                                        job.get('job_publisher'), list) else 'No link available')
                            ),
                            'job_id': job.get('job_id', 'N/A'),
                            'keywords': query
                        }
                        print("DEBUG -> apply_link:", job_info['apply_link'])
                        jobs.append(job_info)
                    return jobs
                else:
                    print(f"No results for keyword: {query}")
            else:
                print(f"API request failed for keyword '{query}' with status code: {response.status_code}")

        print("No job results found for any keyword.")
        return []

    except Exception as e:
        print(f"Error fetching job recommendations: {e}")
        return []
