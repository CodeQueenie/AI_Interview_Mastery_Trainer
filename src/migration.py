"""
Migration Utilities for AI Interview Mastery Trainer

This module provides utilities for importing and exporting session data
between different formats (JSON, SQLite, CSV).

Author: Nicole LeGuern (CodeQueenie)
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
import shutil

# Import the session manager
from src.session_manager import get_session_manager
# Import centralized logging
from src.logger import get_logger, log_function, INFO, DEBUG

# Initialize logger
logger = get_logger("migration")


@log_function(level=INFO)
def import_json_sessions(directory="session_data", user_id="default_user"):
    """
    Import session data from JSON files into the SQLite database.
    
    Args:
        directory (str): Directory containing JSON session files
        user_id (str): User ID to associate with imported sessions
    
    Returns:
        int: Number of sessions imported
    """
    try:
        if not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist")
            return 0
        
        # Get session manager
        session_manager = get_session_manager()
        
        count = 0
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                file_path = os.path.join(directory, filename)
                
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    
                    # Extract data
                    timestamp = data.get("timestamp", datetime.now().isoformat())
                    history = data.get("history", [])
                    scores = data.get("scores", {})
                    
                    # Convert old format scores if needed
                    if isinstance(scores, dict) and any(isinstance(v, list) for v in scores.values()):
                        new_scores = {}
                        for category, score_list in scores.items():
                            if score_list:
                                new_scores[category] = sum(score_list) / len(score_list)
                            else:
                                new_scores[category] = 0
                        scores = new_scores
                    
                    # Save to database
                    session_id = session_manager.save_session(history, scores, user_id)
                    if session_id:
                        count += 1
                        logger.info(f"Imported session from {filename} with ID {session_id}")
                    else:
                        logger.warning(f"Failed to import session from {filename}")
                
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error in {filename}: {e}")
                except KeyError as e:
                    logger.error(f"Missing key in {filename}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error importing {filename}: {e}")
        
        logger.info(f"Successfully imported {count} sessions from {directory}")
        return count
    
    except Exception as e:
        logger.error(f"Error importing JSON sessions: {e}")
        return 0


@log_function(level=INFO)
def export_session_to_json(session_id, output_dir="session_data/exports"):
    """
    Export a session from the SQLite database to a JSON file.
    
    Args:
        session_id (int): ID of the session to export
        output_dir (str): Directory to save the exported file
        
    Returns:
        str: Path to the exported file if successful, None otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get session manager
        session_manager = get_session_manager()
        
        # Load session data
        history, scores = session_manager.load_session(session_id)
        
        if not history:
            logger.warning(f"No history found for session {session_id}")
            return None
        
        # Get session summary for metadata
        summary = session_manager.get_session_summary(session_id)
        
        # Create export data
        export_data = {
            "session_id": session_id,
            "timestamp": summary.get("timestamp") if summary else datetime.now().isoformat(),
            "user_id": summary.get("user_id") if summary else "default_user",
            "history": history,
            "scores": scores
        }
        
        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_id}_{timestamp_str}.json"
        file_path = os.path.join(output_dir, filename)
        
        # Write to file
        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Session {session_id} exported to {file_path}")
        return file_path
    
    except json.JSONEncodeError as e:
        logger.error(f"JSON encoding error exporting session {session_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error exporting session {session_id} to JSON: {e}")
        return None


@log_function(level=INFO)
def export_performance_to_csv(user_id="default_user", output_dir="session_data/exports"):
    """
    Export performance data to a CSV file.
    
    Args:
        user_id (str): User ID
        output_dir (str): Directory to save the exported file
        
    Returns:
        str: Path to the exported file if successful, None otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get session manager
        session_manager = get_session_manager()
        
        # Get performance data
        df = session_manager.get_user_performance(user_id)
        
        if df.empty:
            logger.warning(f"No performance data found for user {user_id}")
            return None
        
        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_{user_id}_{timestamp_str}.csv"
        file_path = os.path.join(output_dir, filename)
        
        # Export to CSV
        df.to_csv(file_path, index=False)
        
        logger.info(f"Performance data for user {user_id} exported to {file_path}")
        return file_path
    
    except pd.errors.EmptyDataError:
        logger.error(f"No data to export for user {user_id}")
        return None
    except Exception as e:
        logger.error(f"Error exporting performance data for user {user_id}: {e}")
        return None


@log_function(level=INFO)
def backup_database(output_dir="session_data/backups"):
    """
    Create a backup of the SQLite database.
    
    Args:
        output_dir (str): Directory to save the backup
        
    Returns:
        str: Path to the backup file if successful, None otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get session manager and database path
        session_manager = get_session_manager()
        db_path = session_manager.db_path
        
        if not os.path.exists(db_path):
            logger.warning(f"Database file {db_path} does not exist")
            return None
        
        # Generate backup filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"sessions_backup_{timestamp_str}.db"
        backup_path = os.path.join(output_dir, backup_filename)
        
        # Create backup using SQLite backup API
        try:
            # Close existing connection
            session_manager.close()
            
            # Copy the database file
            shutil.copy2(db_path, backup_path)
            
            # Verify backup
            if os.path.exists(backup_path) and os.path.getsize(backup_path) > 0:
                logger.info(f"Database backed up to {backup_path}")
                
                # Reconnect to the database
                session_manager._initialize_db()
                
                return backup_path
            else:
                logger.error("Backup verification failed")
                return None
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error during backup: {e}")
            return None
    
    except Exception as e:
        logger.error(f"Error backing up database: {e}")
        return None


@log_function(level=INFO)
def restore_database_from_backup(backup_path, target_path=None):
    """
    Restore the SQLite database from a backup.
    
    Args:
        backup_path (str): Path to the backup file
        target_path (str): Path to restore to (if None, uses the default path)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(backup_path):
            logger.warning(f"Backup file {backup_path} does not exist")
            return False
        
        # Get session manager
        session_manager = get_session_manager()
        
        # Determine target path
        if target_path is None:
            target_path = session_manager.db_path
        
        # Create backup of current database before restoring
        if os.path.exists(target_path):
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            pre_restore_backup = f"{target_path}.pre_restore_{timestamp_str}"
            
            try:
                # Close existing connection
                session_manager.close()
                
                # Create backup of current database
                shutil.copy2(target_path, pre_restore_backup)
                logger.info(f"Created pre-restore backup at {pre_restore_backup}")
                
                # Copy backup to target
                shutil.copy2(backup_path, target_path)
                
                # Verify restore
                if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                    logger.info(f"Database restored from {backup_path} to {target_path}")
                    
                    # Reconnect to the database
                    session_manager._initialize_db()
                    
                    return True
                else:
                    logger.error("Restore verification failed")
                    
                    # Try to restore from pre-restore backup
                    if os.path.exists(pre_restore_backup):
                        shutil.copy2(pre_restore_backup, target_path)
                        logger.info("Reverted to pre-restore state")
                    
                    return False
                
            except sqlite3.Error as e:
                logger.error(f"SQLite error during restore: {e}")
                return False
        else:
            # If target doesn't exist, just copy the backup
            shutil.copy2(backup_path, target_path)
            logger.info(f"Database restored from {backup_path} to {target_path}")
            
            # Reconnect to the database
            session_manager._initialize_db()
            
            return True
    
    except Exception as e:
        logger.error(f"Error restoring database: {e}")
        return False


if __name__ == "__main__":
    # For testing
    import_result = import_json_sessions()
    print(f"Imported {import_result} sessions")
    
    # Test export
    session_manager = get_session_manager()
    recent_sessions = session_manager.get_recent_sessions(limit=1)
    
    if recent_sessions:
        session_id = recent_sessions[0]["id"]
        export_path = export_session_to_json(session_id)
        print(f"Exported session to {export_path}")
        
        # Test performance export
        csv_path = export_performance_to_csv()
        print(f"Exported performance data to {csv_path}")
    
    # Test backup
    backup_path = backup_database()
    print(f"Backed up database to {backup_path}")
    
    # Close connection
    session_manager.close()
