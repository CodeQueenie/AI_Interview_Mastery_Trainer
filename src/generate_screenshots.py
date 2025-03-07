"""
Generate simulated screenshots for the AI Interview Mastery Trainer README.

This script creates visualization images that simulate the appearance of the
application's different modules and dashboard.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Set the style
plt.style.use('ggplot')
sns.set_palette("viridis")

# 1. Dashboard visualization
def create_dashboard_screenshot():
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 3, figure=fig)
    
    # Performance over time
    ax1 = fig.add_subplot(gs[0, :])
    dates = pd.date_range(start='2025-01-01', periods=30)
    performance = np.cumsum(np.random.normal(0.5, 0.3, 30))
    performance = [max(0, min(p, 10)) for p in performance]  # Constrain to 0-10
    ax1.plot(dates, performance, marker='o', linestyle='-', linewidth=2)
    ax1.set_title('Performance Over Time', fontsize=14)
    ax1.set_ylim(0, 10)
    ax1.set_ylabel('Score')
    
    # Category distribution
    ax2 = fig.add_subplot(gs[1, 0:2])
    categories = ['Arrays', 'Trees', 'Dynamic Programming', 'Graphs', 'Sorting']
    values = [8, 6, 4, 7, 9]
    sns.barplot(x=categories, y=values, ax=ax2)
    ax2.set_title('Coding Topics Performance', fontsize=14)
    ax2.set_ylim(0, 10)
    ax2.set_ylabel('Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Strengths and weaknesses
    ax3 = fig.add_subplot(gs[1, 2])
    labels = ['Coding', 'Theory', 'Algorithm Design']
    sizes = [35, 40, 25]
    ax3.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Question Type Distribution', fontsize=14)
    
    # Recent activity
    ax4 = fig.add_subplot(gs[2, :])
    activities = ['Coding Q1', 'Theory Q7', 'Algorithm Design Q2', 'Coding Q5', 'Theory Q3']
    scores = [7, 9, 6, 8, 10]
    sns.barplot(x=activities, y=scores, ax=ax4)
    ax4.set_title('Recent Activity Performance', fontsize=14)
    ax4.set_ylim(0, 10)
    ax4.set_ylabel('Score')
    
    plt.tight_layout()
    plt.savefig('images/dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Coding round interface
def create_coding_round_screenshot():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create a code editor look
    editor_rect = Rectangle((0.05, 0.1), 0.9, 0.8, fill=True, color='#2d2d2d', alpha=0.8)
    ax.add_patch(editor_rect)
    
    # Problem statement
    problem_text = """
    Problem: Two Sum
    
    Given an array of integers nums and an integer target, return indices of the two 
    numbers such that they add up to target.
    
    Example:
    Input: nums = [2, 7, 11, 15], target = 9
    Output: [0, 1]
    Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
    
    Constraints:
    - 2 <= nums.length <= 10^4
    - -10^9 <= nums[i] <= 10^9
    - -10^9 <= target <= 10^9
    - Only one valid answer exists.
    """
    
    # Solution code
    solution_code = """
    # Optimal Solution (O(n) time complexity)
    def two_sum(nums, target):
        # Create a hash map to store values and indices
        num_map = {}  # value -> index
        
        # Iterate through the array once
        for i, num in enumerate(nums):
            # Calculate the complement needed
            complement = target - num
            
            # Check if complement exists in our map
            if complement in num_map:
                # Return both indices
                return [num_map[complement], i]
            
            # Add current number to map
            num_map[num] = i
            
        # No solution found
        return []
    
    # Test with example
    nums = [2, 7, 11, 15]
    target = 9
    result = two_sum(nums, target)
    print(f"Result: {result}")  # Output: [0, 1]
    """
    
    # Add text to the figure
    ax.text(0.07, 0.85, problem_text, color='white', fontsize=10, family='monospace', verticalalignment='top')
    ax.text(0.07, 0.45, solution_code, color='#a6e22e', fontsize=10, family='monospace', verticalalignment='top')
    
    # Add a title
    plt.figtext(0.5, 0.95, "Coding Round: Algorithm Problem", ha="center", fontsize=16, color='black')
    
    # Add execution result
    result_text = "Execution Result: [0, 1] ✓ Correct!"
    plt.figtext(0.5, 0.05, result_text, ha="center", fontsize=12, color='green')
    
    plt.tight_layout()
    plt.savefig('images/coding_round.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Theory round interface
def create_theory_round_screenshot():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create a quiz look
    quiz_rect = Rectangle((0.05, 0.1), 0.9, 0.8, fill=True, color='#f5f5f5', alpha=0.8)
    ax.add_patch(quiz_rect)
    
    # Question
    question_text = """
    Question: What is the difference between L1 and L2 regularization?
    
    Select the correct answer:
    """
    
    # Options
    options = [
        "A) L1 adds absolute values of weights, L2 adds squared weights",
        "B) L1 is for classification, L2 is for regression",
        "C) L1 uses gradient descent, L2 uses stochastic gradient descent",
        "D) L1 prevents underfitting, L2 prevents overfitting"
    ]
    
    # Explanation
    explanation = """
    Explanation: L1 regularization (Lasso) adds the absolute values of the weights to the 
    loss function, which can lead to sparse models as it can drive some weights to exactly zero. 
    L2 regularization (Ridge) adds the squared values of the weights, which penalizes large 
    weights more strongly but rarely sets weights to exactly zero.
    """
    
    # Add text to the figure
    ax.text(0.07, 0.85, question_text, color='black', fontsize=12, family='sans-serif', verticalalignment='top')
    
    # Add options with highlighted correct answer
    for i, option in enumerate(options):
        if i == 0:  # Correct answer
            ax.text(0.07, 0.75 - i*0.07, option, color='green', fontsize=11, family='sans-serif', 
                   verticalalignment='top', weight='bold')
            # Add a checkmark
            ax.text(0.05, 0.75 - i*0.07, "✓", color='green', fontsize=14, family='sans-serif', 
                   verticalalignment='top', weight='bold')
        else:
            ax.text(0.07, 0.75 - i*0.07, option, color='black', fontsize=11, family='sans-serif', 
                   verticalalignment='top')
    
    # Add explanation
    explanation_rect = Rectangle((0.07, 0.35), 0.86, 0.2, fill=True, color='#e6f7ff', alpha=0.5)
    ax.add_patch(explanation_rect)
    ax.text(0.09, 0.53, explanation, color='black', fontsize=10, family='sans-serif', verticalalignment='top')
    
    # Add a title
    plt.figtext(0.5, 0.95, "Theory Round: AI/ML Concepts", ha="center", fontsize=16, color='black')
    
    # Add score
    score_text = "Your Score: 8/10 questions correct"
    plt.figtext(0.5, 0.05, score_text, ha="center", fontsize=12, color='blue')
    
    plt.tight_layout()
    plt.savefig('images/theory_round.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Algorithm design interface
def create_algorithm_design_screenshot():
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    # Problem statement
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    problem_rect = Rectangle((0, 0), 1, 1, fill=True, color='#f5f5f5', alpha=0.8)
    ax1.add_patch(problem_rect)
    problem_text = """
    Problem: Customer Churn Prediction
    
    Design an algorithm to predict customer churn 
    for a subscription-based service. You are 
    provided with historical customer data including 
    demographics, usage patterns, and engagement 
    metrics.
    
    Difficulty: Medium
    Category: Algorithm Design
    """
    ax1.text(0.05, 0.95, problem_text, color='black', fontsize=10, 
             family='sans-serif', verticalalignment='top')
    ax1.set_title('Problem Statement', fontsize=12)
    
    # Step-by-step approach
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    approach_rect = Rectangle((0, 0), 1, 1, fill=True, color='#e6f7ff', alpha=0.8)
    ax2.add_patch(approach_rect)
    approach_text = """
    Step-by-Step Approach:
    
    1. Data preprocessing
       - Handle missing values
       - Normalize features
       - Encode categorical variables
    
    2. Feature engineering & selection
       - Create interaction features
       - Use correlation analysis
       - Apply feature importance
    
    3. Model selection & training
       - Train multiple models
       - Use cross-validation
    
    4. Evaluation & tuning
       - Compare using AUC-ROC
       - Fine-tune hyperparameters
    """
    ax2.text(0.05, 0.95, approach_text, color='black', fontsize=10, 
             family='sans-serif', verticalalignment='top')
    ax2.set_title('Design Approach', fontsize=12)
    
    # Model comparison visualization
    ax3 = fig.add_subplot(gs[1, :])
    models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Neural Network']
    auc_scores = [0.78, 0.85, 0.87, 0.83]
    precision = [0.72, 0.80, 0.82, 0.79]
    recall = [0.65, 0.76, 0.79, 0.74]
    
    x = np.arange(len(models))
    width = 0.25
    
    ax3.bar(x - width, auc_scores, width, label='AUC-ROC')
    ax3.bar(x, precision, width, label='Precision')
    ax3.bar(x + width, recall, width, label='Recall')
    
    ax3.set_ylabel('Score')
    ax3.set_title('Model Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('images/algorithm_design.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating simulated screenshots...")
    create_dashboard_screenshot()
    create_coding_round_screenshot()
    create_theory_round_screenshot()
    create_algorithm_design_screenshot()
    print("Screenshots generated successfully in the 'images' directory.")
