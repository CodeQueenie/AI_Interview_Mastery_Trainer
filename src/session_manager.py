"""
Session Manager for AI Interview Mastery Trainer

This module provides functions for managing user session data using SQLite.
It handles saving, loading, and querying session history and user scores.

Author: Nicole LeGuern (CodeQueenie)
"""

import os
import json
import sqlite3
from datetime import datetime
import pandas as pd

# Import centralized logging
from src.logger import get_logger, log_function, DEBUG, INFO

# Initialize logger
logger = get_logger("session_manager")


class SessionManager:
    """
    Class for managing user session data using SQLite.
    """
    
    def __init__(self, db_path="session_data/sessions.db"):
        """
        Initialize the SessionManager with a database path.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
        # Initialize database
        self._initialize_db()
    
    @log_function(level=DEBUG)
    def _initialize_db(self):
        """
        Initialize the SQLite database with required tables.
        """
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            
            # Create sessions table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT DEFAULT 'default_user'
                )
            """)
            
            # Create session_history table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    question_id INTEGER NOT NULL,
                    question_title TEXT,
                    question_category TEXT,
                    user_answer TEXT,
                    correct_answer TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            # Create user_scores table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    category TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            # Add indexes for faster queries
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_history_timestamp ON session_history(timestamp)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_scores_category ON user_scores(category)")
            
            self.conn.commit()
            logger.info("Database initialized successfully")
        
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.rollback()
            raise
    
    @log_function(level=INFO)
    def save_session(self, session_history, user_scores, user_id="default_user"):
        """
        Save session data to the database.
        
        Args:
            session_history (list): List of session history entries
            user_scores (dict): Dictionary of user scores by category
            user_id (str, optional): User identifier
            
        Returns:
            int: Session ID if successful, None otherwise
        """
        try:
            # Create a new session
            timestamp = datetime.now().isoformat()
            self.cursor.execute(
                "INSERT INTO sessions (timestamp, user_id) VALUES (?, ?)",
                (timestamp, user_id)
            )
            session_id = self.cursor.lastrowid
            
            # Save session history
            for entry in session_history:
                self.cursor.execute(
                    """
                    INSERT INTO session_history 
                    (session_id, timestamp, category, question_id, question_title, 
                    question_category, user_answer, correct_answer)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        entry.get("timestamp", timestamp),
                        entry.get("category", "unknown"),
                        entry.get("question_id", 0),
                        entry.get("question_title", ""),
                        entry.get("question_category", ""),
                        json.dumps(entry.get("user_answer", "")),
                        json.dumps(entry.get("correct_answer", ""))
                    )
                )
            
            # Save user scores
            for category, score in user_scores.items():
                self.cursor.execute(
                    "INSERT INTO user_scores (session_id, category, score, timestamp) VALUES (?, ?, ?, ?)",
                    (session_id, category, score, timestamp)
                )
            
            self.conn.commit()
            logger.info(f"Session {session_id} saved successfully for user {user_id}")
            return session_id
        
        except sqlite3.Error as e:
            logger.error(f"Error saving session: {e}")
            if self.conn:
                self.conn.rollback()
            return None
        except Exception as e:
            logger.error(f"Unexpected error saving session: {e}")
            if self.conn:
                self.conn.rollback()
            return None
    
    @log_function(level=INFO)
    def load_session(self, session_id):
        """
        Load session data from the database.
        
        Args:
            session_id (int): Session ID to load
            
        Returns:
            tuple: (session_history, user_scores) if successful, (None, None) otherwise
        """
        try:
            # Check if session exists
            self.cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
            if not self.cursor.fetchone():
                logger.warning(f"Session {session_id} not found")
                return None, None
            
            # Load session history
            self.cursor.execute(
                "SELECT * FROM session_history WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            history_rows = self.cursor.fetchall()
            
            session_history = []
            for row in history_rows:
                entry = {
                    "id": row[0],
                    "session_id": row[1],
                    "timestamp": row[2],
                    "category": row[3],
                    "question_id": row[4],
                    "question_title": row[5],
                    "question_category": row[6],
                    "user_answer": json.loads(row[7]) if row[7] else "",
                    "correct_answer": json.loads(row[8]) if row[8] else ""
                }
                session_history.append(entry)
            
            # Load user scores
            self.cursor.execute(
                "SELECT category, score FROM user_scores WHERE session_id = ?",
                (session_id,)
            )
            score_rows = self.cursor.fetchall()
            
            user_scores = {}
            for row in score_rows:
                user_scores[row[0]] = row[1]
            
            logger.info(f"Session {session_id} loaded successfully with {len(session_history)} history entries and {len(user_scores)} score entries")
            return session_history, user_scores
        
        except sqlite3.Error as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None, None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error loading session {session_id}: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error loading session {session_id}: {e}")
            return None, None
    
    @log_function(level=INFO)
    def get_recent_sessions(self, limit=10, user_id=None):
        """
        Get a list of recent sessions.
        
        Args:
            limit (int, optional): Maximum number of sessions to return
            user_id (str, optional): Filter by user ID
            
        Returns:
            list: List of session dictionaries
        """
        try:
            query = "SELECT id, timestamp, user_id FROM sessions ORDER BY timestamp DESC LIMIT ?"
            params = (limit,)
            
            if user_id:
                query = "SELECT id, timestamp, user_id FROM sessions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?"
                params = (user_id, limit)
            
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            
            sessions = []
            for row in rows:
                session = {
                    "id": row[0],
                    "timestamp": row[1],
                    "user_id": row[2]
                }
                sessions.append(session)
            
            logger.info(f"Retrieved {len(sessions)} recent sessions")
            return sessions
        
        except sqlite3.Error as e:
            logger.error(f"Error getting recent sessions: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting recent sessions: {e}")
            return []
    
    @log_function(level=INFO)
    def get_session_summary(self, session_id):
        """
        Get a summary of a session.
        
        Args:
            session_id (int): Session ID
            
        Returns:
            dict: Session summary
        """
        try:
            # Get session info
            self.cursor.execute(
                "SELECT timestamp, user_id FROM sessions WHERE id = ?",
                (session_id,)
            )
            session_row = self.cursor.fetchone()
            
            if not session_row:
                logger.warning(f"Session {session_id} not found")
                return None
            
            # Get question counts by category
            self.cursor.execute(
                """
                SELECT category, COUNT(*) FROM session_history 
                WHERE session_id = ? GROUP BY category
                """,
                (session_id,)
            )
            category_rows = self.cursor.fetchall()
            
            categories = {}
            for row in category_rows:
                categories[row[0]] = row[1]
            
            # Get scores
            self.cursor.execute(
                "SELECT category, score FROM user_scores WHERE session_id = ?",
                (session_id,)
            )
            score_rows = self.cursor.fetchall()
            
            scores = {}
            for row in score_rows:
                scores[row[0]] = row[1]
            
            summary = {
                "id": session_id,
                "timestamp": session_row[0],
                "user_id": session_row[1],
                "categories": categories,
                "scores": scores,
                "total_questions": sum(categories.values())
            }
            
            logger.info(f"Generated summary for session {session_id}")
            return summary
        
        except sqlite3.Error as e:
            logger.error(f"Error getting session summary for session {session_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting session summary for session {session_id}: {e}")
            return None
    
    @log_function(level=INFO)
    def get_user_performance(self, user_id="default_user"):
        """
        Get performance data for a user across all sessions.
        
        Args:
            user_id (str, optional): User identifier
            
        Returns:
            pandas.DataFrame: Performance data
        """
        try:
            # Get all sessions for the user
            self.cursor.execute(
                "SELECT id FROM sessions WHERE user_id = ? ORDER BY timestamp",
                (user_id,)
            )
            session_rows = self.cursor.fetchall()
            
            if not session_rows:
                logger.warning(f"No sessions found for user {user_id}")
                return pd.DataFrame()
            
            # Get all scores for these sessions
            session_ids = [row[0] for row in session_rows]
            placeholders = ",".join(["?"] * len(session_ids))
            
            query = f"""
                SELECT s.timestamp, us.category, us.score
                FROM user_scores us
                JOIN sessions s ON us.session_id = s.id
                WHERE s.id IN ({placeholders})
                ORDER BY s.timestamp
            """
            
            self.cursor.execute(query, session_ids)
            score_rows = self.cursor.fetchall()
            
            # Convert to DataFrame
            df = pd.DataFrame(score_rows, columns=["timestamp", "category", "score"])
            
            logger.info(f"Retrieved performance data for user {user_id} with {len(df)} entries")
            return df
        
        except sqlite3.Error as e:
            logger.error(f"Error getting user performance for user {user_id}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error getting user performance for user {user_id}: {e}")
            return pd.DataFrame()
    
    @log_function(level=INFO)
    def get_all_sessions(self, user_id=None):
        """
        Get all sessions for a user or all sessions if user_id is None.
        
        Args:
            user_id (str, optional): User identifier to filter sessions
            
        Returns:
            list: List of session dictionaries with id and timestamp
        """
        try:
            if user_id:
                self.cursor.execute(
                    "SELECT id, timestamp FROM sessions WHERE user_id = ? ORDER BY timestamp DESC",
                    (user_id,)
                )
            else:
                self.cursor.execute(
                    "SELECT id, timestamp FROM sessions ORDER BY timestamp DESC"
                )
            
            session_rows = self.cursor.fetchall()
            
            sessions = []
            for row in session_rows:
                sessions.append({
                    "id": row[0],
                    "timestamp": row[1]
                })
            
            logger.info(f"Retrieved {len(sessions)} sessions for user {user_id if user_id else 'all users'}")
            return sessions
        
        except sqlite3.Error as e:
            logger.error(f"Error getting all sessions for user {user_id if user_id else 'all users'}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting all sessions for user {user_id if user_id else 'all users'}: {e}")
            return []
    
    @log_function(level=INFO)
    def get_recent_activity(self, limit=5, user_id=None):
        """
        Get recent activity across all sessions.
        
        Args:
            limit (int, optional): Maximum number of entries to return
            user_id (str, optional): Filter by user ID
            
        Returns:
            list: List of recent activity entries
        """
        try:
            query = """
                SELECT sh.id, s.id, sh.timestamp, sh.category, sh.question_title, s.user_id
                FROM session_history sh
                JOIN sessions s ON sh.session_id = s.id
                ORDER BY sh.timestamp DESC
                LIMIT ?
            """
            params = (limit,)
            
            if user_id:
                query = """
                    SELECT sh.id, s.id, sh.timestamp, sh.category, sh.question_title, s.user_id
                    FROM session_history sh
                    JOIN sessions s ON sh.session_id = s.id
                    WHERE s.user_id = ?
                    ORDER BY sh.timestamp DESC
                    LIMIT ?
                """
                params = (user_id, limit)
            
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            
            activity = []
            for row in rows:
                entry = {
                    "id": row[0],
                    "session_id": row[1],
                    "timestamp": row[2],
                    "category": row[3],
                    "question_title": row[4],
                    "user_id": row[5]
                }
                activity.append(entry)
            
            logger.info(f"Retrieved {len(activity)} recent activity entries")
            return activity
        
        except sqlite3.Error as e:
            logger.error(f"Error getting recent activity: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting recent activity: {e}")
            return []
    
    def close(self):
        """
        Close the database connection.
        """
        try:
            if self.conn:
                self.conn.close()
                logger.info("Database connection closed")
        except sqlite3.Error as e:
            logger.error(f"Error closing database connection: {e}")
        except Exception as e:
            logger.error(f"Unexpected error closing database connection: {e}")


# Singleton instance
_session_manager = None

def initialize_session_state():
    """
    Initialize Streamlit session state variables if they don't exist.
    This ensures all required state variables are available throughout the application.
    """
    import streamlit as st
    
    # User scores and performance tracking
    if "user_scores" not in st.session_state:
        st.session_state.user_scores = {"coding": [], "theory": [], "algorithm_design": []}
    
    # Current question tracking
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    
    if "current_category" not in st.session_state:
        st.session_state.current_category = None
    
    # Answer state
    if "show_answer" not in st.session_state:
        st.session_state.show_answer = False
    
    if "user_answer" not in st.session_state:
        st.session_state.user_answer = ""
    
    # Session history
    if "session_history" not in st.session_state:
        st.session_state.session_history = []
    
    # UI state
    if "dashboard_tab" not in st.session_state:
        st.session_state.dashboard_tab = "progress"
    
    # User identification
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default_user"
    
    if "db_session_id" not in st.session_state:
        st.session_state.db_session_id = None
    
    # Hints on demand
    if "hint_index" not in st.session_state:
        st.session_state.hint_index = 0
    
    # In-memory leaderboard (in production, store and retrieve from DB)
    if "leaderboard" not in st.session_state:
        st.session_state.leaderboard = []  # e.g. [{"user":"...", "score":..., "efficiency":...}, ...]

@log_function(level=INFO)
def get_session_manager():
    """
    Get the singleton SessionManager instance.
    
    Returns:
        SessionManager: The session manager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


if __name__ == "__main__":
    # For testing
    sm = get_session_manager()
    
    # Test saving a session
    session_history = [
        {
            "category": "coding",
            "question_id": 1,
            "question_title": "Two Sum",
            "question_category": "Arrays",
            "user_answer": "def two_sum(nums, target):\n    # Implementation",
            "correct_answer": "def two_sum(nums, target):\n    # Optimal implementation"
        }
    ]
    
    user_scores = {
        "coding": 80,
        "theory": 90,
        "algorithm_design": 75
    }
    
    session_id = sm.save_session(session_history, user_scores)
    print(f"Saved session with ID: {session_id}")
    
    # Test loading a session
    loaded_history, loaded_scores = sm.load_session(session_id)
    print(f"Loaded session with {len(loaded_history)} history entries and {len(loaded_scores)} score entries")
    
    # Test getting recent sessions
    recent_sessions = sm.get_recent_sessions(limit=5)
    print(f"Recent sessions: {recent_sessions}")
    
    # Close connection
    sm.close()
