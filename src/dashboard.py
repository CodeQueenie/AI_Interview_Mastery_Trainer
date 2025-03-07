"""
Dashboard module for the AI Interview Mastery Trainer application.

This module provides functions for creating and displaying the dashboard
with user progress and performance metrics.

Author: Nicole LeGuern (CodeQueenie)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import from src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils import generate_performance_summary
    from src.visualizations import (
        plot_category_distribution,
        plot_theory_performance,
        plot_activity_heatmap,
        plot_radar_chart,
        plot_score_distribution
    )
except ImportError as e:
    st.error(f"Error importing dashboard dependencies: {e}")
    st.stop()


def display_dashboard(session_history, user_scores):
    """
    Display the dashboard with user progress and performance metrics.
    
    Args:
        session_history (list): List of session history entries
        user_scores (dict): Dictionary of user scores by category
    """
    try:
        st.header("Your Progress Dashboard")
        
        if not session_history:
            st.info("You haven't attempted any questions yet. Start practicing to see your progress!")
            return
        
        # Generate performance summary
        performance_summary = generate_performance_summary(session_history)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Questions Attempted",
                value=performance_summary["total_questions"]
            )
        
        with col2:
            theory_entries = [entry for entry in session_history if entry["category"] == "theory"]
            if theory_entries:
                correct_count = sum(1 for entry in theory_entries if entry.get("user_answer") == entry.get("correct_answer"))
                accuracy = correct_count / len(theory_entries) * 100
                st.metric(
                    label="Theory Questions Accuracy",
                    value=f"{accuracy:.1f}%"
                )
            else:
                st.metric(
                    label="Theory Questions Accuracy",
                    value="N/A"
                )
        
        with col3:
            # Calculate streak (consecutive days with activity)
            if len(session_history) > 0:
                dates = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S").date() for entry in session_history]
                unique_dates = sorted(set(dates), reverse=True)
                
                streak = 1
                for i in range(1, len(unique_dates)):
                    if (unique_dates[i-1] - unique_dates[i]).days == 1:
                        streak += 1
                    else:
                        break
                
                st.metric(
                    label="Current Streak",
                    value=f"{streak} day{'s' if streak != 1 else ''}"
                )
            else:
                st.metric(
                    label="Current Streak",
                    value="0 days"
                )
        
        # Display strengths and areas for improvement
        st.subheader("Performance Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Strengths")
            if performance_summary["strengths"]:
                for strength in performance_summary["strengths"]:
                    st.markdown(f"- {strength}")
            else:
                st.markdown("Keep practicing to identify your strengths!")
        
        with col2:
            st.markdown("#### Areas for Improvement")
            if performance_summary["areas_for_improvement"]:
                for area in performance_summary["areas_for_improvement"]:
                    st.markdown(f"- {area}")
            else:
                st.markdown("Keep practicing to identify areas for improvement!")
        
        # Display visualizations
        st.subheader("Visualizations")
        
        # Category distribution
        st.markdown("#### Question Category Distribution")
        category_chart = plot_category_distribution(session_history)
        if category_chart:
            st.image(f"data:image/png;base64,{category_chart}")
        else:
            st.info("Not enough data to generate category distribution chart.")
        
        # Theory performance
        if user_scores["theory"]:
            st.markdown("#### Theory Questions Performance")
            theory_chart = plot_theory_performance(user_scores["theory"])
            if theory_chart:
                st.image(f"data:image/png;base64,{theory_chart}")
            else:
                st.info("Not enough data to generate theory performance chart.")
        
        # Activity heatmap
        if len(session_history) >= 3:
            st.markdown("#### Activity Heatmap")
            heatmap = plot_activity_heatmap(session_history)
            if heatmap:
                st.image(f"data:image/png;base64,{heatmap}")
            else:
                st.info("Not enough data to generate activity heatmap.")
        
        # Radar chart
        if performance_summary["categories"]:
            st.markdown("#### Performance Radar Chart")
            radar_chart = plot_radar_chart(performance_summary)
            if radar_chart:
                st.image(f"data:image/png;base64,{radar_chart}")
            else:
                st.info("Not enough data to generate radar chart.")
        
        # Recent activity
        st.subheader("Recent Activity")
        if session_history:
            recent_history = session_history[-10:]  # Show last 10 entries
            
            # Create DataFrame for display
            df = pd.DataFrame([
                {
                    "Timestamp": entry["timestamp"],
                    "Category": entry["category"].replace("_", " ").title(),
                    "Question": entry.get("question_title", entry.get("question_category", f"Question {entry.get('question_id', '')}")),
                    "Result": "Correct" if entry.get("user_answer") == entry.get("correct_answer") else "Attempted" if "user_answer" in entry else "N/A"
                }
                for entry in reversed(recent_history)
            ])
            
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No activity recorded yet.")
        
        # Save session button
        if st.button("Save Dashboard as PDF"):
            st.warning("This feature is not yet implemented. Please check back later!")
    
    except Exception as e:
        st.error(f"Error displaying dashboard: {e}")


def display_recommendations(session_history):
    """
    Display personalized recommendations based on the user's performance.
    
    Args:
        session_history (list): List of session history entries
    """
    try:
        st.header("Personalized Recommendations")
        
        if not session_history:
            st.info("Start practicing to receive personalized recommendations!")
            return
        
        # Generate performance summary
        performance_summary = generate_performance_summary(session_history)
        
        # Count questions by category
        category_counts = {}
        for entry in session_history:
            category = entry["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Identify least practiced category
        if len(category_counts) < 3:
            missing_categories = set(["coding", "theory", "algorithm_design"]) - set(category_counts.keys())
            if missing_categories:
                least_practiced = list(missing_categories)[0]
            else:
                least_practiced = min(category_counts.items(), key=lambda x: x[1])[0]
        else:
            least_practiced = min(category_counts.items(), key=lambda x: x[1])[0]
        
        # Theory accuracy
        theory_entries = [entry for entry in session_history if entry["category"] == "theory"]
        theory_accuracy = 0
        if theory_entries:
            correct_count = sum(1 for entry in theory_entries if entry.get("user_answer") == entry.get("correct_answer"))
            theory_accuracy = correct_count / len(theory_entries) * 100
        
        # Generate recommendations
        recommendations = []
        
        # Category balance recommendation
        if least_practiced:
            recommendations.append({
                "title": f"Practice more {least_practiced.replace('_', ' ').title()} questions",
                "description": f"You've focused less on {least_practiced.replace('_', ' ').title()} questions. Balancing your practice across all categories will help you develop a well-rounded skill set for AI interviews."
            })
        
        # Theory accuracy recommendation
        if theory_entries:
            if theory_accuracy < 70:
                recommendations.append({
                    "title": "Review AI/ML Theory Concepts",
                    "description": "Your accuracy on theory questions is below 70%. Consider reviewing fundamental AI/ML concepts to improve your understanding."
                })
        else:
            recommendations.append({
                "title": "Start practicing Theory Questions",
                "description": "You haven't attempted any theory questions yet. These questions test your understanding of AI/ML fundamentals, which are crucial for AI interviews."
            })
        
        # Consistency recommendation
        dates = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S").date() for entry in session_history]
        unique_dates = set(dates)
        
        if len(unique_dates) < 3 and len(session_history) > 5:
            recommendations.append({
                "title": "Practice more consistently",
                "description": "Regular practice is key to mastering interview skills. Try to practice a little bit every day rather than cramming all at once."
            })
        
        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations):
                st.markdown(f"#### {i+1}. {rec['title']}")
                st.markdown(rec["description"])
                st.markdown("---")
        else:
            st.info("Keep practicing to receive more personalized recommendations!")
        
        # Study resources
        st.subheader("Recommended Study Resources")
        
        # Coding resources
        st.markdown("#### Algorithm & Data Structures")
        st.markdown("""
        - [LeetCode](https://leetcode.com/) - Company-specific problem sets
        - [HackerRank](https://www.hackerrank.com/) - Structured learning paths
        - "Cracking the Coding Interview" by Gayle Laakmann McDowell
        - [Udemy's Data Structures & Algorithms Course](https://www.udemy.com/course/data-structures-and-algorithms-deep-dive-using-java/)
        """)
        
        # Theory resources
        st.markdown("#### AI/ML Theory")
        st.markdown("""
        - [Andrew Ng's Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) (Coursera)
        - "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
        - [StatQuest YouTube Channel](https://www.youtube.com/c/joshstarmer)
        """)
        
        # Algorithm design resources
        st.markdown("#### Algorithm Design for ML")
        st.markdown("""
        - [Kaggle](https://www.kaggle.com/) - Competitions and notebook examples
        - [CodeBasics YouTube Channel](https://www.youtube.com/c/codebasics)
        - [freeCodeCamp ML and Deep Learning Tutorials](https://www.freecodecamp.org/news/tag/machine-learning/)
        """)
    
    except Exception as e:
        st.error(f"Error displaying recommendations: {e}")


if __name__ == "__main__":
    # This is for testing the dashboard module independently
    st.title("AI Interview Mastery Trainer - Dashboard")
    
    # Sample data for testing
    sample_history = [
        {
            "timestamp": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "category": "theory",
            "question_id": 1,
            "question_category": "Machine Learning Fundamentals",
            "user_answer": "B",
            "correct_answer": "B"
        },
        {
            "timestamp": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "category": "theory",
            "question_id": 2,
            "question_category": "Machine Learning Algorithms",
            "user_answer": "A",
            "correct_answer": "B"
        },
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "category": "coding",
            "question_id": 1,
            "question_title": "Two Sum",
            "user_answer": {"brute_force": "...", "optimal_solution": "..."}
        }
    ]
    
    sample_scores = {
        "coding": [],
        "theory": [1, 0],
        "algorithm_design": []
    }
    
    display_dashboard(sample_history, sample_scores)
    display_recommendations(sample_history)
