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
    # Import study guides for tracking progress
    from data.study_guides import STUDY_GUIDES
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


def display_study_progress():
    """
    Display the user's progress through study guides and learning materials.
    Allows users to mark topics as completed and track their learning journey.
    """
    st.header(" Study Progress Tracker")
    
    # Initialize study progress in session state if not already present
    if 'study_progress' not in st.session_state:
        st.session_state.study_progress = {
            'coding': {},
            'theory': {},
            'algorithm_design': {}
        }
    
    # Create tabs for different categories
    tab1, tab2, tab3 = st.tabs(["Coding Concepts", "Theory & ML", "Algorithm Design"])
    
    with tab1:
        st.subheader("Coding Concepts Progress")
        display_category_progress('coding')
    
    with tab2:
        st.subheader("Theory & ML Concepts Progress")
        display_category_progress('theory')
    
    with tab3:
        st.subheader("Algorithm Design Progress")
        display_category_progress('algorithm_design')
    
    # Overall progress
    st.subheader("Overall Learning Progress")
    
    # Calculate overall progress
    total_topics = 0
    completed_topics = 0
    
    for category in ['coding', 'theory', 'algorithm_design']:
        for topic in STUDY_GUIDES.get(category, {}):
            if topic != 'default':  # Skip default guide
                total_topics += 1
                if st.session_state.study_progress[category].get(topic, {}).get('completed', False):
                    completed_topics += 1
    
    if total_topics > 0:
        progress_percentage = (completed_topics / total_topics) * 100
        st.progress(progress_percentage / 100)
        st.markdown(f"**{completed_topics}/{total_topics} topics completed ({progress_percentage:.1f}%)**")
        
        # Gamification elements
        if completed_topics > 0:
            if completed_topics == total_topics:
                st.balloons()
                st.success(" Congratulations! You've completed all study topics! You're ready to ace your interviews!")
            elif progress_percentage >= 75:
                st.success(" Amazing progress! You're well on your way to interview mastery!")
            elif progress_percentage >= 50:
                st.info(" Great job! You're making solid progress in your interview preparation!")
            elif progress_percentage >= 25:
                st.info(" Good start! Keep going with your learning journey!")
            else:
                st.info(" You've begun your learning journey! Keep studying to improve your skills!")
    else:
        st.info("No study topics available. Check back later for updated content!")


def display_category_progress(category):
    """
    Display progress for a specific category of study guides.
    
    Args:
        category (str): The category to display ('coding', 'theory', or 'algorithm_design')
    """
    if category not in STUDY_GUIDES or not STUDY_GUIDES[category]:
        st.info(f"No study guides available for {category} yet.")
        return
    
    # Filter out the default guide
    topics = [topic for topic in STUDY_GUIDES[category] if topic != 'default']
    
    if not topics:
        st.info(f"No specific topics available for {category} yet.")
        return
    
    # Create a nice grid layout for topic cards
    cols = st.columns(3)
    
    for i, topic in enumerate(topics):
        # Get or initialize topic progress
        if topic not in st.session_state.study_progress[category]:
            st.session_state.study_progress[category][topic] = {
                'completed': False,
                'last_reviewed': None,
                'notes': ''
            }
        
        topic_progress = st.session_state.study_progress[category][topic]
        
        # Display topic card in the appropriate column
        with cols[i % 3]:
            with st.container(border=True):
                # Topic header with completion status
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{topic}**")
                with col2:
                    completed = topic_progress['completed']
                    if completed:
                        st.markdown("✅")
                    else:
                        st.markdown("⏳")
                
                # Last reviewed date
                if topic_progress['last_reviewed']:
                    st.caption(f"Last reviewed: {topic_progress['last_reviewed']}")
                else:
                    st.caption("Not yet reviewed")
                
                # Actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📖 Review", key=f"review_{category}_{topic}"):
                        st.session_state.study_progress[category][topic]['last_reviewed'] = datetime.now().strftime("%Y-%m-%d")
                        # Set a flag to navigate to the study guide
                        st.session_state.navigate_to_study_guide = {
                            'category': category,
                            'topic': topic
                        }
                        st.rerun()
                
                with col2:
                    if st.button(
                        "✓ Done" if not completed else "🔄 Redo", 
                        key=f"complete_{category}_{topic}"
                    ):
                        st.session_state.study_progress[category][topic]['completed'] = not completed
                        st.rerun()
                
                # Notes section
                with st.expander("📝 Notes"):
                    notes = st.text_area(
                        "Your notes:",
                        value=topic_progress['notes'],
                        key=f"notes_{category}_{topic}",
                        height=100
                    )
                    if notes != topic_progress['notes']:
                        st.session_state.study_progress[category][topic]['notes'] = notes
                        st.success("Notes saved!")


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
        
        # Study recommendations based on performance
        st.subheader(" Recommended Study Topics")
        
        # Get areas for improvement
        areas_for_improvement = performance_summary.get("areas_for_improvement", [])
        
        if areas_for_improvement:
            st.markdown("Based on your performance, we recommend focusing on these topics:")
            
            for i, area in enumerate(areas_for_improvement[:3]):  # Show top 3 recommendations
                topic_category = area.split(" (")[0] if " (" in area else area
                
                # Find the appropriate broad category
                broad_category = None
                for category in ["coding", "theory", "algorithm_design"]:
                    if topic_category in STUDY_GUIDES.get(category, {}):
                        broad_category = category
                        break
                
                if not broad_category:
                    # Try to match by substring
                    for category in ["coding", "theory", "algorithm_design"]:
                        for guide_topic in STUDY_GUIDES.get(category, {}):
                            if topic_category.lower() in guide_topic.lower() or guide_topic.lower() in topic_category.lower():
                                broad_category = category
                                topic_category = guide_topic
                                break
                        if broad_category:
                            break
                
                if broad_category:
                    with st.container(border=True):
                        st.markdown(f"**{i+1}. {topic_category}**")
                        st.markdown(f"*Category: {broad_category.replace('_', ' ').title()}*")
                        
                        # Show a snippet from the study guide
                        guide = STUDY_GUIDES.get(broad_category, {}).get(topic_category)
                        if guide:
                            st.markdown(f"**Summary:** {guide['summary'][:150]}...")
                        
                        if st.button("Review This Topic", key=f"rec_review_{i}"):
                            # Set a flag to navigate to the study guide
                            st.session_state.navigate_to_study_guide = {
                                'category': broad_category,
                                'topic': topic_category
                            }
                            st.rerun()
                else:
                    st.markdown(f"**{i+1}. {topic_category}**")
                    st.markdown("*No specific study guide available for this topic yet.*")
        else:
            st.info("Keep practicing to receive personalized study recommendations!")
        
        # Practice recommendations
        st.subheader(" Recommended Practice")
        
        # Recommend question types based on activity
        question_types = ["coding", "theory", "algorithm_design"]
        question_counts = {qt: 0 for qt in question_types}
        
        for entry in session_history:
            if entry["category"] in question_counts:
                question_counts[entry["category"]] += 1
        
        # Find least practiced question type
        least_practiced = min(question_counts.items(), key=lambda x: x[1])
        
        if least_practiced[1] == 0:
            st.markdown(f"You haven't tried any **{least_practiced[0].replace('_', ' ').title()}** questions yet!")
            st.markdown("These questions will help you develop your skills in this important area.")
            
            if st.button(f"Try a {least_practiced[0].replace('_', ' ').title()} Question", key="try_least_practiced"):
                st.session_state.navigate_to_question_type = least_practiced[0]
                st.rerun()
        else:
            # Recommend based on performance
            if performance_summary["areas_for_improvement"]:
                area = performance_summary["areas_for_improvement"][0]
                st.markdown(f"Practice more questions in **{area.split(' (')[0]}** to improve your skills.")
                
                # Determine the question type from the area
                question_type = "coding"
                if "ML" in area or "Neural" in area or "Supervised" in area:
                    question_type = "theory"
                elif "System" in area or "Design" in area:
                    question_type = "algorithm_design"
                
                if st.button(f"Practice {area.split(' (')[0]} Questions", key="practice_improvement_area"):
                    st.session_state.navigate_to_question_type = question_type
                    st.session_state.navigate_to_category = area.split(' (')[0]
                    st.rerun()
        
        # Learning streak
        st.subheader(" Learning Streak")
        
        # Calculate streak
        if len(session_history) > 0:
            dates = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S").date() for entry in session_history]
            unique_dates = sorted(set(dates), reverse=True)
            
            streak = 1
            for i in range(1, len(unique_dates)):
                if (unique_dates[i-1] - unique_dates[i]).days == 1:
                    streak += 1
                else:
                    break
            
            st.markdown(f"**Current streak: {streak} day{'s' if streak != 1 else ''}**")
            
            # Gamification elements
            if streak >= 7:
                st.success(" Wow! A week-long streak! Your consistency is impressive!")
            elif streak >= 3:
                st.info(" Nice 3+ day streak! Keep up the good work!")
            else:
                st.info(" You're building momentum! Come back tomorrow to continue your streak!")
            
            # Next milestone
            next_milestone = 3 if streak < 3 else 7 if streak < 7 else 14 if streak < 14 else 30
            days_to_go = next_milestone - streak
            
            if days_to_go > 0:
                st.markdown(f"**Next milestone: {next_milestone} days (just {days_to_go} more day{'s' if days_to_go != 1 else ''}!)**")
                st.progress(streak / next_milestone)
        else:
            st.info("Start practicing today to begin your learning streak!")
            st.markdown("**Consistency is key to interview success!**")
    
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
    display_study_progress()
    display_recommendations(sample_history)
