"""
Utility functions for the AI Interview Mastery Trainer application.

This module provides helper functions for evaluating user solutions,
generating feedback, and analyzing code complexity.

Author: Nicole LeGuern (CodeQueenie)
"""

import re
import ast
import numpy as np
from collections import Counter

# Import centralized logging
from src.logger import get_logger, log_function, INFO

# Initialize logger
logger = get_logger("utils")


@log_function(level=INFO)
def analyze_code_complexity(code_str):
    """
    Analyze the time and space complexity of a given code snippet.
    This is a simplified analysis and may not be accurate for all cases.
    
    Args:
        code_str (str): The code string to analyze
        
    Returns:
        tuple: (time_complexity, space_complexity) as strings
    """
    try:
        # Default complexities
        time_complexity = "Unknown"
        space_complexity = "Unknown"
        
        # Check for common patterns
        if re.search(r"for\s+\w+\s+in\s+range\(\w+\):\s*for\s+\w+\s+in\s+range\(\w+\)", code_str):
            # Nested loops suggest O(n²)
            time_complexity = "O(n²)"
        elif re.search(r"for\s+\w+\s+in\s+range\(\w+\)", code_str):
            # Single loop suggests O(n)
            time_complexity = "O(n)"
        elif "while" in code_str and "=" in code_str:
            # While loop might be O(n) or O(log n)
            if "/" in code_str or "*" in code_str:
                time_complexity = "O(log n)"
            else:
                time_complexity = "O(n)"
        else:
            # No loops might suggest O(1)
            time_complexity = "O(1)"
        
        # Check for space complexity
        if "append" in code_str or "extend" in code_str or "[" in code_str:
            # Creating lists suggests O(n) space
            space_complexity = "O(n)"
        elif "{" in code_str or "dict(" in code_str or "set(" in code_str:
            # Creating dictionaries or sets suggests O(n) space
            space_complexity = "O(n)"
        else:
            # Otherwise, assume constant space
            space_complexity = "O(1)"
        
        logger.debug(f"Analyzed code complexity: Time={time_complexity}, Space={space_complexity}")
        return time_complexity, space_complexity
    
    except Exception as e:
        logger.error(f"Error analyzing code complexity: {e}")
        return "Unknown", "Unknown"


@log_function(level=INFO)
def evaluate_coding_solution(user_solution, optimal_solution):
    """
    Evaluate a user's coding solution against the optimal solution.
    
    Args:
        user_solution (dict): User's solution with code and complexity analysis
        optimal_solution (dict): Optimal solution with code and complexity analysis
        
    Returns:
        dict: Feedback on the user's solution
    """
    try:
        feedback = {
            "correctness": 0,  # 0-100 score
            "efficiency": 0,   # 0-100 score
            "style": 0,        # 0-100 score
            "comments": []     # List of feedback comments
        }
        
        # Analyze user's code complexity
        user_time, user_space = analyze_code_complexity(user_solution["optimal_solution"])
        
        # Compare with provided complexity
        if user_solution["time_complexity"].lower() == optimal_solution["time_complexity"].lower():
            feedback["comments"].append("✅ Your time complexity analysis is correct.")
            feedback["efficiency"] += 50
        else:
            feedback["comments"].append(f"❓ Your time complexity analysis ({user_solution['time_complexity']}) differs from the optimal solution ({optimal_solution['time_complexity']}). Consider reviewing your approach.")
        
        if user_solution["space_complexity"].lower() == optimal_solution["space_complexity"].lower():
            feedback["comments"].append("✅ Your space complexity analysis is correct.")
            feedback["efficiency"] += 50
        else:
            feedback["comments"].append(f"❓ Your space complexity analysis ({user_solution['space_complexity']}) differs from the optimal solution ({optimal_solution['space_complexity']}). Consider reviewing your approach.")
        
        # Check for key elements in the solution
        # This is a simplified check and could be enhanced with more sophisticated analysis
        optimal_tokens = set(re.findall(r'\b\w+\b', optimal_solution["code"].lower()))
        user_tokens = set(re.findall(r'\b\w+\b', user_solution["optimal_solution"].lower()))
        
        common_tokens = optimal_tokens.intersection(user_tokens)
        token_similarity = len(common_tokens) / len(optimal_tokens) if optimal_tokens else 0
        
        # Score based on token similarity
        feedback["correctness"] = int(token_similarity * 100)
        
        # Add comments based on correctness score
        if feedback["correctness"] >= 90:
            feedback["comments"].append("✅ Your solution contains most of the key elements of the optimal solution.")
        elif feedback["correctness"] >= 70:
            feedback["comments"].append("✅ Your solution contains many key elements of the optimal solution, but there may be room for improvement.")
        else:
            feedback["comments"].append("❓ Your solution differs significantly from the optimal solution. Consider reviewing the approach.")
        
        # Check code style
        style_score = 100
        
        # Check for comments
        if not re.search(r'#.*', user_solution["optimal_solution"]):
            feedback["comments"].append("💡 Consider adding comments to explain your code.")
            style_score -= 20
        
        # Check for consistent indentation
        if re.search(r'^\s+\S+', user_solution["optimal_solution"], re.MULTILINE):
            if not re.search(r'^\s{4}\S+', user_solution["optimal_solution"], re.MULTILINE):
                feedback["comments"].append("💡 Consider using consistent indentation (4 spaces recommended).")
                style_score -= 20
        
        # Check for meaningful variable names
        short_vars = re.findall(r'\b[a-z]{1}\b', user_solution["optimal_solution"])
        if len(short_vars) > 3:
            feedback["comments"].append("💡 Consider using more descriptive variable names instead of single-letter variables.")
            style_score -= 20
        
        feedback["style"] = style_score
        
        logger.info(f"Evaluated coding solution with correctness={feedback['correctness']}, efficiency={feedback['efficiency']}, style={feedback['style']}")
        return feedback
    
    except Exception as e:
        logger.error(f"Error evaluating coding solution: {e}")
        return {
            "correctness": 0,
            "efficiency": 0,
            "style": 0,
            "comments": ["Error evaluating solution"]
        }


@log_function(level=INFO)
def evaluate_theory_answer(user_answer, correct_answer):
    """
    Evaluate a user's answer to a theory question.
    
    Args:
        user_answer (str): User's selected answer (e.g., 'A', 'B', 'C', 'D')
        correct_answer (str): The correct answer
        
    Returns:
        dict: Feedback on the user's answer
    """
    try:
        is_correct = user_answer == correct_answer
        
        feedback = {
            "is_correct": is_correct,
            "score": 100 if is_correct else 0,
            "comments": []
        }
        
        if is_correct:
            feedback["comments"].append(f"✅ Correct! The answer is {correct_answer}.")
        else:
            feedback["comments"].append(f"❌ Incorrect. The correct answer is {correct_answer}.")
        
        logger.info(f"Evaluated theory answer: user={user_answer}, correct={correct_answer}, result={feedback['is_correct']}")
        return feedback
    
    except Exception as e:
        logger.error(f"Error evaluating theory answer: {e}")
        return {
            "is_correct": False,
            "score": 0,
            "comments": ["Error evaluating answer"]
        }


@log_function(level=INFO)
def evaluate_algorithm_design(user_design, optimal_design):
    """
    Evaluate a user's algorithm design against the optimal design.
    
    Args:
        user_design (dict): User's design approach and pseudocode
        optimal_design (dict): Optimal design approach and pseudocode
        
    Returns:
        dict: Feedback on the user's design
    """
    try:
        feedback = {
            "approach_score": 0,  # 0-100 score
            "pseudocode_score": 0,  # 0-100 score
            "overall_score": 0,  # 0-100 score
            "comments": []  # List of feedback comments
        }
        
        # Extract key phrases from the optimal approach
        optimal_approach = optimal_design["explanation"].lower()
        optimal_pseudocode = optimal_design["pseudocode"].lower()
        
        user_approach = user_design["approach"].lower()
        user_pseudocode = user_design["pseudocode"].lower()
        
        # Check for key elements in the approach
        key_phrases = [
            "preprocess", "clean", "normalize", "feature", "train", "test", "split",
            "cross-validation", "evaluate", "metric", "hyperparameter", "tune",
            "regularization", "overfitting", "underfitting"
        ]
        
        # Count key phrases in both approaches
        optimal_phrase_count = sum(1 for phrase in key_phrases if phrase in optimal_approach)
        user_phrase_count = sum(1 for phrase in key_phrases if phrase in user_approach)
        
        # Calculate approach score
        if optimal_phrase_count > 0:
            approach_similarity = min(user_phrase_count / optimal_phrase_count, 1.0)
            feedback["approach_score"] = int(approach_similarity * 100)
        
        # Check for key steps in the pseudocode
        optimal_steps = [line.strip() for line in optimal_pseudocode.split("\n") if line.strip()]
        user_steps = [line.strip() for line in user_pseudocode.split("\n") if line.strip()]
        
        # Simple step count comparison
        step_ratio = min(len(user_steps) / len(optimal_steps), 1.0) if optimal_steps else 0
        feedback["pseudocode_score"] = int(step_ratio * 100)
        
        # Calculate overall score
        feedback["overall_score"] = (feedback["approach_score"] + feedback["pseudocode_score"]) // 2
        
        # Add comments based on scores
        if feedback["approach_score"] >= 80:
            feedback["comments"].append("✅ Your approach covers most of the key considerations in the optimal solution.")
        elif feedback["approach_score"] >= 60:
            feedback["comments"].append("✅ Your approach covers many important aspects, but there may be room for improvement.")
        else:
            feedback["comments"].append("❓ Your approach may be missing some key considerations. Review the optimal approach for guidance.")
        
        if feedback["pseudocode_score"] >= 80:
            feedback["comments"].append("✅ Your pseudocode includes most of the essential steps.")
        elif feedback["pseudocode_score"] >= 60:
            feedback["comments"].append("✅ Your pseudocode includes many important steps, but could be more comprehensive.")
        else:
            feedback["comments"].append("❓ Your pseudocode may be missing some key steps. Review the optimal solution for guidance.")
        
        logger.info(f"Evaluated algorithm design with approach_score={feedback['approach_score']}, pseudocode_score={feedback['pseudocode_score']}, overall_score={feedback['overall_score']}")
        return feedback
    
    except Exception as e:
        logger.error(f"Error evaluating algorithm design: {e}")
        return {
            "approach_score": 0,
            "pseudocode_score": 0,
            "overall_score": 0,
            "comments": ["Error evaluating design"]
        }


@log_function(level=INFO)
def validate_question(question, question_type):
    """
    Validate that a question dictionary has all required fields based on its type.
    
    Args:
        question (dict): Question dictionary to validate
        question_type (str): Type of question ('coding', 'theory', or 'algorithm_design')
        
    Returns:
        tuple: (is_valid, error_message) - Boolean indicating if valid and error message if not
    """
    try:
        # Common required fields for all question types
        if "id" not in question:
            return False, "Question is missing 'id' field"
            
        # Specific validation based on question type
        if question_type == "coding":
            required_fields = ["title", "difficulty", "category", "problem", "examples", "constraints", "hints"]
            for field in required_fields:
                if field not in question:
                    return False, f"Coding question is missing '{field}' field"
                    
        elif question_type == "theory":
            required_fields = ["category", "question", "options", "correct_answer", "explanation"]
            for field in required_fields:
                if field not in question:
                    return False, f"Theory question is missing '{field}' field"
            
            # Validate options is a list
            if not isinstance(question["options"], list):
                return False, "Theory question options must be a list"
                
        elif question_type == "algorithm_design":
            required_fields = ["title", "difficulty", "category", "problem", "hints"]
            for field in required_fields:
                if field not in question:
                    return False, f"Algorithm design question is missing '{field}' field"
            
            # Validate design_approach if present
            if "design_approach" in question:
                if not isinstance(question["design_approach"], dict):
                    return False, "Algorithm design question 'design_approach' must be a dictionary"
        else:
            return False, f"Unknown question type: {question_type}"
            
        return True, ""
    except Exception as e:
        return False, str(e)


@log_function(level=INFO)
def generate_performance_summary(session_history):
    """
    Generate a summary of the user's performance based on their session history.
    
    Args:
        session_history (list): List of session history entries
        
    Returns:
        dict: Performance summary statistics
    """
    try:
        if not session_history:
            return {
                "total_questions": 0,
                "categories": {},
                "strengths": [],
                "areas_for_improvement": []
            }
        
        summary = {
            "total_questions": len(session_history),
            "categories": {},
            "strengths": [],
            "areas_for_improvement": []
        }
        
        # Count questions by category
        category_counts = Counter([entry["category"] for entry in session_history])
        
        for category, count in category_counts.items():
            summary["categories"][category] = {
                "count": count,
                "percentage": count / summary["total_questions"] * 100
            }
        
        # Analyze theory questions (where we have clear correct/incorrect answers)
        theory_entries = [entry for entry in session_history if entry["category"] == "theory"]
        if theory_entries:
            correct_count = sum(1 for entry in theory_entries if entry.get("user_answer") == entry.get("correct_answer"))
            accuracy = correct_count / len(theory_entries) * 100
            
            summary["categories"]["theory"]["accuracy"] = accuracy
            
            if accuracy >= 80:
                summary["strengths"].append("Strong understanding of AI/ML theory concepts")
            elif accuracy <= 50:
                summary["areas_for_improvement"].append("Review core AI/ML theoretical concepts")
        
        # Determine most practiced areas
        if summary["categories"]:
            most_practiced = max(summary["categories"].items(), key=lambda x: x[1]["count"])
            least_practiced = min(summary["categories"].items(), key=lambda x: x[1]["count"])
            
            if most_practiced[1]["count"] > 3:
                summary["strengths"].append(f"Good focus on {most_practiced[0]} questions")
            
            if least_practiced[1]["count"] < 2 and summary["total_questions"] > 5:
                summary["areas_for_improvement"].append(f"Practice more {least_practiced[0]} questions")
        
        logger.info(f"Generated performance summary: total_questions={summary['total_questions']}, strengths={summary['strengths']}, areas_for_improvement={summary['areas_for_improvement']}")
        return summary
    
    except Exception as e:
        logger.error(f"Error generating performance summary: {e}")
        return {
            "total_questions": len(session_history) if session_history else 0,
            "categories": {},
            "strengths": [],
            "areas_for_improvement": ["Error generating performance summary"]
        }
