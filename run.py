"""
Run script for AI Interview Mastery Trainer

This script launches the Streamlit application for the AI Interview Mastery Trainer.

Usage:
    python run.py

Author: Nicole LeGuern (CodeQueenie)
"""

import os
import sys
import subprocess
import webbrowser
import logging
import sqlite3
from time import sleep
from datetime import datetime


def setup_logging():
    """
    Set up logging configuration.
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    log_filename = f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("ai_interview_trainer")


def check_database():
    """
    Check if the SQLite database exists and is properly initialized.
    If not, initialize it.
    
    Returns:
        bool: True if database is ready, False otherwise
    """
    logger = logging.getLogger("ai_interview_trainer")
    
    try:
        # Create session_data directory if it doesn't exist
        os.makedirs("session_data", exist_ok=True)
        
        db_path = "session_data/sessions.db"
        
        # Check if database file exists
        if not os.path.exists(db_path):
            logger.info(f"Database file {db_path} does not exist. It will be created when the app starts.")
            return True
        
        # Check if database is valid
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if required tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('sessions', 'session_history', 'user_scores')")
        tables = cursor.fetchall()
        
        if len(tables) < 3:
            logger.warning(f"Database {db_path} is missing some required tables. It will be reinitialized when the app starts.")
        else:
            logger.info(f"Database {db_path} is valid and ready to use.")
        
        conn.close()
        return True
    
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return False


def main():
    """
    Main function to run the Streamlit application.
    """
    # Set up logging
    logger = setup_logging()
    
    logger.info("Starting AI Interview Mastery Trainer...")
    
    # Check if required directories exist
    if not os.path.exists("data"):
        logger.info("Creating data directory...")
        os.makedirs("data", exist_ok=True)
    
    if not os.path.exists("session_data"):
        logger.info("Creating session_data directory...")
        os.makedirs("session_data", exist_ok=True)
    
    if not os.path.exists("session_data/exports"):
        logger.info("Creating session_data/exports directory...")
        os.makedirs("session_data/exports", exist_ok=True)
    
    if not os.path.exists("session_data/backups"):
        logger.info("Creating session_data/backups directory...")
        os.makedirs("session_data/backups", exist_ok=True)
    
    # Check if required files exist
    required_files = [
        "data/coding_questions.py",
        "data/theory_questions.py",
        "data/algorithm_design_questions.py",
        "src/app.py",
        "src/utils.py",
        "src/visualizations.py",
        "src/dashboard.py",
        "src/session_manager.py",
        "src/migration.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error("The following required files are missing:")
        for file in missing_files:
            logger.error(f"  - {file}")
        logger.error("Please make sure all required files are present before running the application.")
        return
    
    # Check database
    if not check_database():
        logger.error("Database check failed. Please check the logs for more information.")
        return
    
    # Run the Streamlit application
    try:
        logger.info("Launching Streamlit application...")
        cmd = ["streamlit", "run", "src/app.py"]
        
        # Start the process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for the server to start
        url = None
        for line in process.stdout:
            line = line.strip()
            logger.info(f"Streamlit: {line}")
            
            if "You can now view your Streamlit app in your browser" in line:
                # Extract the URL from the output
                for line in process.stdout:
                    line = line.strip()
                    logger.info(f"Streamlit: {line}")
                    if "Local URL:" in line:
                        url = line.split("Local URL:")[1].strip()
                        logger.info(f"Opening browser at {url}")
                        webbrowser.open(url)
                        break
                break
        
        if not url:
            logger.warning("Could not extract local URL from Streamlit output.")
        
        # Keep the process running and log any errors
        while True:
            error_line = process.stderr.readline().strip()
            if error_line:
                logger.error(f"Streamlit error: {error_line}")
            
            # Check if process is still running
            if process.poll() is not None:
                logger.info(f"Streamlit process exited with code {process.returncode}")
                break
            
            sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("Shutting down the application...")
        if 'process' in locals() and process.poll() is None:
            process.terminate()
            logger.info("Streamlit process terminated.")
    except Exception as e:
        logger.exception(f"Error running the application: {e}")


if __name__ == "__main__":
    main()
