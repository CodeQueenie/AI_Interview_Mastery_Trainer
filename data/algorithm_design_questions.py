"""
Merged and Enhanced Algorithm Design Questions for AI Interview Mastery Trainer

This module combines the best elements of both previous versions:
- Uses the expanded difficulty levels and question variety.
- Retains visualization capabilities for model performance analysis.
- Maintains detailed evaluation criteria and key considerations.
- Adds an interactive step-by-step walkthrough functionality.

Author: Nicole LeGuern (CodeQueenie)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st

def plot_model_comparison(model_scores, model_names):
    """
    Generate a bar chart comparing model performances.
    
    Args:
        model_scores (list): List of model evaluation scores.
        model_names (list): List of model names.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=model_names, y=model_scores, palette="viridis", ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Evaluation Score")
    st.pyplot(fig)

ALGORITHM_DESIGN_QUESTIONS = [
    {
        "id": 1,
        "title": "Feature Selection for Predictive Modeling",
        "difficulty": "Easy",
        "category": "Algorithm Design",
        "problem": (
            "Design an algorithm to select the most relevant features for a predictive model. "
            "You are given a dataset with multiple features, some of which may be irrelevant or redundant."
        ),
        "hints": [
            "Consider feature importance methods like correlation analysis, mutual information, or decision tree-based importance.",
            "Regularization techniques like L1 (Lasso) regression can help in feature selection.",
            "Check for multicollinearity using variance inflation factor (VIF)."
        ],
        "design_approach": {
            "explanation": (
                "First, perform exploratory data analysis to understand feature distributions. "
                "Next, use correlation analysis or statistical tests to remove redundant features. "
                "Regularization techniques like L1 regression or decision tree-based feature importance can help further refine feature selection."
            ),
            "pseudocode": (
                "1. Load dataset\n"
                "2. Compute correlation matrix to remove highly correlated features\n"
                "3. Use Lasso regression to assign importance to features\n"
                "4. Train model with selected features and evaluate performance\n"
                "5. Compare different feature selection techniques and finalize the model"
            ),
            "key_considerations": (
                "Feature selection should improve model interpretability without significantly reducing accuracy. "
                "Regularization techniques help prevent overfitting, and multicollinearity should be addressed to avoid redundant features."
            ),
            "evaluation_criteria": (
                "The effectiveness of feature selection should be assessed based on predictive performance. "
                "Compare different techniques using cross-validation and feature importance metrics."
            )
        }
    },
    {
        "id": 2,
        "title": "Time-Series Forecasting for Sales Prediction",
        "difficulty": "Medium",
        "category": "Algorithm Design",
        "problem": (
            "Design an algorithm to forecast future sales based on historical data. "
            "The dataset contains daily sales records, and the goal is to predict sales for the next 30 days."
        ),
        "hints": [
            "Consider models like ARIMA, Exponential Smoothing, or LSTMs for forecasting.",
            "Time-series data often requires transformations like differencing or seasonal adjustments.",
            "Cross-validation for time-series differs from standard cross-validation due to temporal dependencies."
        ],
        "design_approach": {
            "explanation": (
                "Preprocess the data by handling missing values and applying transformations like logarithm scaling. "
                "Use a rolling-window approach for cross-validation. Train a forecasting model such as ARIMA or an LSTM-based model, "
                "and evaluate predictions using RMSE and MAPE."
            ),
            "pseudocode": (
                "1. Load and preprocess sales data\n"
                "2. Apply time-series transformations (log scale, differencing)\n"
                "3. Split data into training and validation sets using rolling windows\n"
                "4. Train forecasting model (ARIMA/LSTM)\n"
                "5. Evaluate performance using RMSE and MAPE\n"
                "6. Tune hyperparameters and generate future predictions"
            )
        }
    },
    {
        "id": 3,
        "title": "Reinforcement Learning for Game AI",
        "difficulty": "Hard",
        "category": "Algorithm Design",
        "problem": (
            "Design an AI agent using Reinforcement Learning to play a game optimally. "
            "The environment provides a reward signal based on the agent’s actions, and the goal is to maximize long-term rewards."
        ),
        "hints": [
            "Consider using Q-learning or Deep Q-Networks (DQN) for decision-making.",
            "Reinforcement learning requires exploration-exploitation trade-offs (e.g., epsilon-greedy).",
            "Train the agent using experience replay and target networks."
        ],
        "design_approach": {
            "explanation": (
                "Define the state space and action space. Implement a reinforcement learning algorithm like Q-learning or DQN. "
                "Train the agent using experience replay and periodically update the target network. Evaluate performance based on cumulative rewards."
            ),
            "pseudocode": (
                "1. Define state space, action space, and reward function\n"
                "2. Initialize Q-table or deep Q-network\n"
                "3. Implement epsilon-greedy strategy for action selection\n"
                "4. Train using experience replay and reward feedback\n"
                "5. Evaluate performance using cumulative rewards"
            )
        }
    }
]

if __name__ == "__main__":
    try:
        # Simple interactive loop to display questions and design approaches
        for question in ALGORITHM_DESIGN_QUESTIONS:
            print("Question ID:", question["id"])
            print("Title:", question["title"])
            print("Difficulty:", question["difficulty"])
            print("Category:", question["category"])
            print("\nProblem Statement:\n", question["problem"])
            print("\nHints:")
            for hint in question["hints"]:
                print("- " + hint)
            print("\nOptimal Design Approach:")
            design = question["design_approach"]
            print("Explanation:\n", design["explanation"])
            print("Pseudocode:\n", design["pseudocode"])
            print("-" * 80 + "\n")
    except Exception as e:
        print("An error occurred while displaying the algorithm design questions:", e)
