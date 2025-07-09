from dotenv import load_dotenv
import os


load_dotenv(dotenv_path=".env")


class Config:
    # Get the SECRET_KEY from environment variable or raise an error if not set
    SECRET_KEY = os.getenv('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY is not set in the environment variables")

    # Set the SQLAlchemy database URI from environment variable, default to a local SQLite if not set
    SQLALCHEMY_DATABASE_URI = os.getenv('SQLALCHEMY_DATABASE_URI', 'sqlite:///db.sqlite3')

    # SQLAlchemy track modifications flag (set to False by default to save resources)
    SQLALCHEMY_TRACK_MODIFICATIONS = os.getenv('SQLALCHEMY_TRACK_MODIFICATIONS', False)

    RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')  # Store API Key in .env
    RAPIDAPI_HOST = os.getenv('RAPIDAPI_HOST', 'jsearch.p.rapidapi.com')  # Default host

    if not RAPIDAPI_KEY:
        raise ValueError("RAPIDAPI_KEY is not set in the environment variables")