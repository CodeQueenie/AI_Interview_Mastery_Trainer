"""
Visualization utilities for the AI Interview Mastery Trainer application.

This module provides functions for creating visualizations of user performance
and progress data using matplotlib and seaborn.

Author: Nicole LeGuern (CodeQueenie)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime, timedelta


def get_img_as_base64(fig):
    """
    Convert a matplotlib figure to a base64 encoded string for embedding in HTML.
    
    Args:
        fig (matplotlib.figure.Figure): The figure to convert
        
    Returns:
        str: Base64 encoded string of the figure
    """
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        return img_str
    except Exception as e:
        print(f"Error converting figure to base64: {e}")
        return ""


def plot_category_distribution(session_history):
    """
    Create a pie chart showing the distribution of questions by category.
    
    Args:
        session_history (list): List of session history entries
        
    Returns:
        str: Base64 encoded string of the figure
    """
    try:
        if not session_history:
            return None
        
        # Count questions by category
        category_counts = {}
        for entry in session_history:
            category = entry["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            "Category": list(category_counts.keys()),
            "Count": list(category_counts.values())
        })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create pie chart
        colors = sns.color_palette("pastel")[0:len(category_counts)]
        ax.pie(
            df["Count"],
            labels=df["Category"],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops={"edgecolor": "white", "linewidth": 1}
        )
        ax.set_title("Distribution of Questions by Category", fontsize=14, pad=20)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_aspect("equal")
        
        return get_img_as_base64(fig)
    
    except Exception as e:
        print(f"Error creating category distribution plot: {e}")
        return None


def plot_theory_performance(user_scores):
    """
    Create a line chart showing the user's performance on theory questions over time.
    
    Args:
        user_scores (list): List of scores (1 for correct, 0 for incorrect)
        
    Returns:
        str: Base64 encoded string of the figure
    """
    try:
        if not user_scores:
            return None
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            "Question": range(1, len(user_scores) + 1),
            "Score": user_scores,
            "Cumulative Accuracy": [sum(user_scores[:i+1]) / (i+1) * 100 for i in range(len(user_scores))]
        })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot cumulative accuracy
        sns.lineplot(
            data=df,
            x="Question",
            y="Cumulative Accuracy",
            marker="o",
            markersize=8,
            linewidth=2,
            color="#4CAF50",
            ax=ax
        )
        
        # Plot individual scores
        for i, score in enumerate(user_scores):
            color = "#4CAF50" if score == 1 else "#F44336"
            ax.scatter(i + 1, df["Cumulative Accuracy"][i], color=color, s=100, zorder=5)
        
        # Customize plot
        ax.set_title("Theory Questions Performance", fontsize=14, pad=20)
        ax.set_xlabel("Question Number", fontsize=12)
        ax.set_ylabel("Cumulative Accuracy (%)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_ylim(0, 105)
        
        # Add horizontal line at 100%
        ax.axhline(y=100, color="#BDBDBD", linestyle="--", alpha=0.5)
        
        # Customize ticks
        ax.set_xticks(range(1, len(user_scores) + 1))
        
        return get_img_as_base64(fig)
    
    except Exception as e:
        print(f"Error creating theory performance plot: {e}")
        return None


def plot_activity_heatmap(session_history):
    """
    Create a heatmap showing the user's activity over time.
    
    Args:
        session_history (list): List of session history entries
        
    Returns:
        str: Base64 encoded string of the figure
    """
    try:
        if not session_history:
            return None
        
        # Extract timestamps and convert to datetime
        timestamps = [datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S") for entry in session_history]
        
        # Get the date range
        if len(timestamps) > 1:
            start_date = min(timestamps).date()
            end_date = max(timestamps).date()
        else:
            start_date = timestamps[0].date()
            end_date = start_date
        
        # Ensure at least a week of data
        if (end_date - start_date).days < 7:
            start_date = end_date - timedelta(days=6)
        
        # Create a date range
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        
        # Count activities by date and category
        activity_counts = {}
        for entry in session_history:
            date = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S").date()
            category = entry["category"]
            
            if date not in activity_counts:
                activity_counts[date] = {}
            
            activity_counts[date][category] = activity_counts[date].get(category, 0) + 1
        
        # Create DataFrame for plotting
        data = []
        for date in date_range:
            date_key = date.date()
            if date_key in activity_counts:
                for category, count in activity_counts[date_key].items():
                    data.append({
                        "Date": date_key,
                        "Category": category,
                        "Count": count
                    })
            else:
                # Add zero counts for all categories
                for category in ["coding", "theory", "algorithm_design"]:
                    data.append({
                        "Date": date_key,
                        "Category": category,
                        "Count": 0
                    })
        
        df = pd.DataFrame(data)
        
        # Pivot the DataFrame for the heatmap
        pivot_df = df.pivot(index="Category", columns="Date", values="Count").fillna(0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Create heatmap
        sns.heatmap(
            pivot_df,
            cmap="YlGnBu",
            linewidths=0.5,
            linecolor="white",
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Number of Questions"},
            ax=ax
        )
        
        # Customize plot
        ax.set_title("Activity Heatmap", fontsize=14, pad=20)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Category", fontsize=12)
        
        # Format date labels
        date_labels = [date.strftime("%Y-%m-%d") for date in date_range]
        ax.set_xticklabels(date_labels, rotation=45, ha="right")
        
        # Format category labels
        category_labels = [cat.replace("_", " ").title() for cat in pivot_df.index]
        ax.set_yticklabels(category_labels, rotation=0)
        
        plt.tight_layout()
        
        return get_img_as_base64(fig)
    
    except Exception as e:
        print(f"Error creating activity heatmap: {e}")
        return None


def plot_radar_chart(performance_summary):
    """
    Create a radar chart showing the user's strengths and weaknesses.
    
    Args:
        performance_summary (dict): Performance summary data
        
    Returns:
        str: Base64 encoded string of the figure
    """
    try:
        if not performance_summary or not performance_summary["categories"]:
            return None
        
        # Define the categories and metrics
        categories = []
        values = []
        
        # Add theory accuracy if available
        if "theory" in performance_summary["categories"] and "accuracy" in performance_summary["categories"]["theory"]:
            categories.append("Theory Knowledge")
            values.append(performance_summary["categories"]["theory"]["accuracy"])
        
        # Add category percentages
        for category, data in performance_summary["categories"].items():
            if category == "theory":
                continue  # Already added above
            
            categories.append(f"{category.replace('_', ' ').title()} Practice")
            values.append(min(data["percentage"], 100))
        
        # Ensure we have at least 3 categories for a meaningful radar chart
        if len(categories) < 3:
            # Add placeholder categories if needed
            while len(categories) < 3:
                categories.append(f"Metric {len(categories) + 1}")
                values.append(0)
        
        # Number of variables
        N = len(categories)
        
        # Create angles for each category
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        
        # Close the polygon
        values.append(values[0])
        angles.append(angles[0])
        categories.append(categories[0])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, linestyle="solid", color="#4CAF50")
        ax.fill(angles, values, alpha=0.25, color="#4CAF50")
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1], fontsize=10)
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        
        # Add grid lines
        ax.set_rgrids([20, 40, 60, 80, 100], angle=0, fontsize=8)
        
        # Set title
        ax.set_title("Performance Radar Chart", fontsize=14, pad=20)
        
        return get_img_as_base64(fig)
    
    except Exception as e:
        print(f"Error creating radar chart: {e}")
        return None


def plot_score_distribution(session_history):
    """
    Create a histogram showing the distribution of scores.
    
    Args:
        session_history (list): List of session history entries
        
    Returns:
        str: Base64 encoded string of the figure
    """
    try:
        # Filter theory questions where we have clear scores
        theory_entries = [entry for entry in session_history if entry["category"] == "theory"]
        
        if not theory_entries:
            return None
        
        # Calculate scores (1 for correct, 0 for incorrect)
        scores = [1 if entry.get("user_answer") == entry.get("correct_answer") else 0 for entry in theory_entries]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create histogram
        sns.histplot(
            scores,
            bins=[0, 0.5, 1],
            kde=False,
            discrete=True,
            shrink=0.8,
            color="#4CAF50",
            ax=ax
        )
        
        # Customize plot
        ax.set_title("Theory Questions Score Distribution", fontsize=14, pad=20)
        ax.set_xlabel("Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Incorrect", "Correct"])
        
        return get_img_as_base64(fig)
    
    except Exception as e:
        print(f"Error creating score distribution plot: {e}")
        return None
