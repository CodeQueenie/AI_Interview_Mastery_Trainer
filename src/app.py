"""
AI Interview Mastery Trainer - Main Application

Author: Nicole LeGuern (CodeQueenie)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import json
import time
from datetime import datetime

# Add the parent directory to the path to import from data directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import question datasets
    from data.coding_questions import CODING_QUESTIONS
    from data.theory_questions import THEORY_QUESTIONS
    from data.algorithm_design_questions import ALGORITHM_DESIGN_QUESTIONS
    
    # Import study guides
    from data.study_guides import STUDY_GUIDES
    
    # Import utility modules
    from src.utils import evaluate_coding_solution, evaluate_theory_answer, evaluate_algorithm_design, validate_question
    from src.utils import generate_preparation_guide, generate_question_checklist, generate_pattern_recognition_tips
    from src.visualizations import plot_category_distribution, plot_theory_performance
    from src.dashboard import display_dashboard, display_recommendations, display_study_progress
    
    # Import session management modules
    from src.session_manager import get_session_manager, initialize_session_state
    from src.migration import export_session_to_json, export_performance_to_csv, backup_database
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# ===============================
# Study Guide Display
# ===============================
def display_study_guide(topic_category, broad_category):
    """
    Display relevant study materials from study_guides.py
    
    Args:
        topic_category (str): The specific topic (e.g., "Arrays & Hashing", "Neural Networks")
        broad_category (str): The broad category ("coding", "theory", "algorithm_design")
    """
    st.markdown(f"## 📚 Study Guide: {topic_category}")
    
    # Try to find an exact match first
    guide = STUDY_GUIDES.get(broad_category, {}).get(topic_category)
    
    # If no exact match, try to find a related topic
    if not guide:
        for topic, content in STUDY_GUIDES.get(broad_category, {}).items():
            if topic_category.lower() in topic.lower() or topic.lower() in topic_category.lower():
                guide = content
                break
    
    # If still no match, use the default guide for the category
    if not guide and "default" in STUDY_GUIDES.get(broad_category, {}):
        guide = STUDY_GUIDES[broad_category]["default"]
    
    # If no guide is available at all, create a custom guide based on the topic
    if not guide:
        # Create a custom study guide based on the topic name and category
        if broad_category == "coding":
            guide = create_coding_guide(topic_category)
        elif broad_category == "theory":
            guide = create_theory_guide(topic_category)
        elif broad_category == "algorithm_design":
            guide = create_algorithm_guide(topic_category)
    
    # Display the guide content
    st.markdown("### 📝 Summary")
    st.write(guide["summary"])
    
    if "key_points" in guide:
        st.markdown("### 🔑 Key Points")
        for point in guide["key_points"]:
            st.markdown(f"- {point}")
    
    if "use_cases" in guide:
        st.markdown("### 🎯 Common Use Cases")
        st.write(", ".join(guide["use_cases"]))
    
    if "example" in guide:
        st.markdown("### 💻 Example Code")
        st.code(guide["example"], language="python")
    
    if "fun_fact" in guide:
        st.markdown("### 🎉 Fun Fact")
        st.info(guide["fun_fact"])
    
    # Display a pre-question checklist
    st.markdown("### ✅ Pre-Question Checklist")
    checklist = generate_question_checklist(broad_category)
    for item in checklist:
        st.markdown(f"{item}")
    
    # If applicable, show pattern recognition tips - FIXED to avoid nested expanders
    if broad_category == "coding" and topic_category in ["Arrays & Hashing", "Dynamic Programming", "Graphs"]:
        st.markdown("### 🧩 Pattern Recognition Tips")
        pattern_tips = generate_pattern_recognition_tips(topic_category)
        
        # Display all pattern tips without using expanders
        for pattern, tips in pattern_tips.items():
            st.markdown(f"**{pattern}**")
            st.markdown(f"*How to recognize:* {tips['recognition']}")
            st.markdown(f"*Approach:* {tips['approach']}")
            st.markdown(f"*Example problems:* {tips['example']}")
            st.markdown("---")

def create_coding_guide(topic):
    """Create a coding-specific study guide with practical interview strategies."""
    return {
        "summary": f"This guide will help you tackle {topic} coding questions effectively in interviews.",
        "key_points": [
            f"Break down {topic} problems into smaller, manageable steps",
            "Start with a brute force solution, then optimize",
            "Consider time and space complexity in your approach",
            "Test your solution with multiple examples, including edge cases",
            "Practice explaining your thought process clearly as you code"
        ],
        "use_cases": ["Technical interviews", "Whiteboard coding", "Take-home assignments"],
        "example": f"""
# Approach for solving {topic} problems:

1. UNDERSTAND THE PROBLEM
   - Clarify any ambiguities with the interviewer
   - Identify input/output formats and constraints
   - Write down test cases including edge cases

2. PLAN YOUR APPROACH
   - Consider appropriate data structures for {topic}
   - Think about algorithmic patterns that might apply
   - Sketch pseudocode before writing actual code

3. IMPLEMENT YOUR SOLUTION
   - Write clean, readable code
   - Use meaningful variable names
   - Add comments for complex logic

4. TEST YOUR SOLUTION
   - Trace through with your test cases
   - Check edge cases (empty input, large input, etc.)
   - Optimize if needed and explain trade-offs
        """,
        "fun_fact": f"In technical interviews, candidates who explain their {topic} solution process clearly are often rated higher than those who just code silently, even if both solutions work correctly!"
    }

def create_theory_guide(topic):
    """Create a theory-specific study guide with practical interview strategies."""
    return {
        "summary": f"This guide will help you answer {topic} theory questions confidently in interviews.",
        "key_points": [
            f"Understand the core concepts of {topic} thoroughly",
            "Be able to explain concepts in simple terms with analogies",
            "Know the advantages, limitations, and trade-offs",
            "Connect theoretical concepts to practical applications",
            "Prepare for follow-up questions that test deeper understanding"
        ],
        "use_cases": ["Technical interviews", "System design discussions", "ML/AI role interviews"],
        "example": f"""
# Approach for answering {topic} theory questions:

1. STRUCTURE YOUR ANSWER
   - Start with a clear, concise definition
   - Explain the key components or principles
   - Provide a real-world example or application
   - Discuss advantages and limitations

2. DEMONSTRATE DEPTH
   - Compare with alternative approaches
   - Explain when this concept is most appropriate
   - Discuss recent advancements if relevant

3. CONNECT TO PRACTICE
   - Explain how you've applied this concept
   - Discuss implementation considerations
   - Mention common pitfalls and how to avoid them

4. HANDLE FOLLOW-UPS
   - Listen carefully to follow-up questions
   - Be honest if you don't know something
   - Think out loud to show your reasoning process
        """,
        "fun_fact": f"Interviewers often ask about {topic} not just to test knowledge, but to see how you communicate complex ideas - a crucial skill for technical roles!"
    }

def create_algorithm_guide(topic):
    """Create an algorithm design-specific study guide with practical interview strategies."""
    return {
        "summary": f"This guide will help you design and explain {topic} algorithms effectively in interviews.",
        "key_points": [
            f"Understand the problem requirements for {topic} thoroughly",
            "Consider multiple approaches before selecting one",
            "Analyze time and space complexity of your solution",
            "Identify and handle edge cases and constraints",
            "Communicate your design decisions and trade-offs clearly"
        ],
        "use_cases": ["System design interviews", "Algorithm optimization tasks", "ML pipeline design"],
        "example": f"""
# Approach for {topic} algorithm design questions:

1. CLARIFY REQUIREMENTS
   - Ask questions to understand the problem scope
   - Identify constraints (time, memory, scalability)
   - Define success criteria and expected outputs

2. EXPLORE APPROACHES
   - Consider multiple solutions with different trade-offs
   - Sketch high-level designs for each approach
   - Evaluate pros and cons of each design

3. DETAIL YOUR SOLUTION
   - Break down into components or steps
   - Explain data flow and processing logic
   - Discuss implementation considerations
   - Analyze complexity (time, space, scalability)

4. EVALUATE AND REFINE
   - Test with example scenarios
   - Identify potential bottlenecks
   - Discuss how your solution could be improved or scaled
        """,
        "fun_fact": f"The best algorithm designers don't just focus on correctness - they balance multiple factors like readability, maintainability, and performance, just as you should in your {topic} solutions!"
    }

# ==================================
# Display Logic: Coding Questions
# ==================================
def display_hints(hints):
    """
    Display a series of progressive hints to help the user solve the problem.
    
    Args:
        hints (list): A list of hint strings, ordered from subtle to more direct
    """
    st.markdown("### 💡 Need a hint?")
    st.markdown("*Hints are progressively more revealing. Try to solve with minimal hints!*")
    
    for i, hint in enumerate(hints):
        hint_key = f"hint_{i+1}"
        with st.expander(f"Hint {i+1}", expanded=False):
            st.markdown(hint)
            if i < len(hints) - 1:
                st.caption("*Still stuck? Try the next hint for more guidance.*")
            else:
                st.caption("*This is the most direct hint. You've got this!*")

def display_coding_question(question):
    """
    Enhanced coding question display:
      - Hints on demand
      - Collapsible problem statements
      - Code highlighting
      - Leaderboard updates
      - Friendly, encouraging UI
    """
    st.title("🧩 Coding Round: Algorithmic Problem Solving")
    st.subheader(f"{question['title']} ({question['difficulty']})")
    st.markdown(f"*Category: {question['category']}*")
    
    # Study Guide Button
    with st.expander("📚 Prepare for this question type", expanded=False):
        st.markdown("### Review study materials before attempting the question")
        st.markdown("Understanding the underlying concepts will help you solve the problem more effectively!")
        
        if st.button("📖 Review Study Guide", key=f"study_guide_btn_{question['id']}"):
            display_study_guide(question['category'], "coding")
    
    # Pre-Question Checklist
    with st.expander("✅ Pre-Question Checklist", expanded=False):
        st.markdown("### Before you start coding, check these off:")
        checklist = generate_question_checklist("coding")
        for item in checklist:
            st.markdown(f"{item}")
    
    # Collapsible problem info
    with st.expander("📝 Problem Breakdown (Click to expand)"):
        st.markdown("### 🎯 Your Mission")
        st.markdown(question["problem"])
        st.markdown("### 🔍 Examples")
        st.markdown(f"```\n{question['examples']}\n```")
        st.markdown("### 📏 Constraints")
        st.markdown(question["constraints"])
        
        # Quick tips for approaching the problem
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        * Break down the problem into smaller steps
        * Think about edge cases - what could go wrong?
        * Start with a simple solution, then optimize
        * Remember: it's okay if your first approach isn't perfect!
        """)
    
    # Hints on Demand
    if "hints" in question:
        display_hints(question["hints"])
    
    # User input
    st.markdown("### 🤔 Your Brute Force Approach")
    st.markdown("*Don't worry about efficiency yet - just get something working!*")
    brute_force_approach = st.text_area("Describe your initial approach:", key="brute_force_input", height=150)
    
    st.markdown("### ✨ Your Optimized Solution")
    st.markdown("*Now let's make it shine! How can you make your solution faster or use less memory?*")
    optimal_solution = st.text_area("Your optimized code:", key="optimal_solution_input", height=300)
    
    col1, col2 = st.columns(2)
    with col1:
        time_complexity = st.text_input("Time Complexity (e.g., O(n)):", key="time_complexity_input")
        st.caption("*Tip: Think about how your solution scales with input size*")
    with col2:
        space_complexity = st.text_input("Space Complexity (e.g., O(1)):", key="space_complexity_input")
        st.caption("*Tip: Consider all the extra memory your solution uses*")
    
    if st.button("✅ Submit Solution", use_container_width=True):
        if not brute_force_approach or not optimal_solution:
            st.warning("🤔 Hmm, looks like you're missing either your brute force approach or optimized solution. Let's fill those in!")
        else:
            st.session_state.user_answer = {
                "brute_force": brute_force_approach,
                "optimal_solution": optimal_solution,
                "time_complexity": time_complexity,
                "space_complexity": space_complexity
            }
            st.session_state.show_answer = True
            
            st.session_state.session_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "category": "coding",
                "question_id": question["id"],
                "question_title": question["title"],
                "question_category": question["category"],
                "user_answer": st.session_state.user_answer,
                "correct_answer": question["optimal_solution"]["code"]
            })
            
            save_session_data()
    
    if st.session_state.show_answer:
        st.markdown("---")
        st.markdown("## 🎓 Solution Breakdown")
        
        st.markdown("### 🔨 Brute Force Approach")
        st.markdown(question["brute_force"]["explanation"])
        st.code(question["brute_force"]["code"])
        st.markdown(f"⏱️ **Time Complexity**: {question['brute_force']['time_complexity']}")
        st.markdown(f"🧠 **Space Complexity**: {question['brute_force']['space_complexity']}")
        
        st.markdown("### 💎 Optimal Solution")
        st.markdown(question["optimal_solution"]["explanation"])
        st.code(question["optimal_solution"]["code"])
        st.markdown(f"⏱️ **Time Complexity**: {question['optimal_solution']['time_complexity']}")
        st.markdown(f"🧠 **Space Complexity**: {question['optimal_solution']['space_complexity']}")
        
        st.markdown("### 🔍 Feedback on Your Solution")
        
        feedback = evaluate_coding_solution(
            st.session_state.user_answer,
            question["optimal_solution"]
        )
        
        # More encouraging feedback
        for comment in feedback["comments"]:
            if "differs" in comment or "Consider" in comment:
                st.info("💡 " + comment)
            else:
                st.success("✅ " + comment)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correctness", f"{feedback['correctness']}%")
            if feedback['correctness'] >= 80:
                st.markdown("🎉 Great job on correctness!")
            elif feedback['correctness'] >= 50:
                st.markdown("👍 You're on the right track!")
            else:
                st.markdown("💪 Keep practicing, you'll get it!")
                
        with col2:
            st.metric("Efficiency", f"{feedback['efficiency']}%")
            if feedback['efficiency'] >= 80:
                st.markdown("🚀 Super efficient solution!")
            elif feedback['efficiency'] >= 50:
                st.markdown("⚡ Good optimization work!")
            else:
                st.markdown("🔧 Let's work on optimizing!")
                
        with col3:
            st.metric("Style", f"{feedback['style']}%")
            if feedback['style'] >= 80:
                st.markdown("✨ Beautiful, clean code!")
            elif feedback['style'] >= 50:
                st.markdown("📝 Nice coding style!")
            else:
                st.markdown("📚 Let's improve readability!")
        
        # Fun fact or joke to lighten the mood
        coding_jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs! 🐛",
            "A programmer had a problem. They thought, 'I know, I'll solve it with threads!' has Now problems. two they 🧵",
            "Why do programmers always mix up Christmas and Halloween? Because Oct 31 == Dec 25! 🎃🎄",
            "The best thing about a Boolean is that even if you're wrong, you're only off by a bit! 0️⃣1️⃣",
            "Why don't programmers like nature? It has too many bugs and no debugging tool! 🐞"
        ]
        import random
        st.markdown(f"### 😄 Random Coding Joke")
        st.markdown(random.choice(coding_jokes))
        
        if st.button("🚀 Next Question", key="next_coding", use_container_width=True):
            # Find the next question in the list
            current_id = st.session_state.current_question["id"]
            next_question = None
            
            # Get all valid coding questions
            valid_questions = []
            for q in CODING_QUESTIONS:
                is_valid, _ = validate_question(q, "coding")
                if is_valid:
                    valid_questions.append(q)
            
            # Sort by ID
            valid_questions.sort(key=lambda x: x["id"])
            
            # Find the next question
            for i, q in enumerate(valid_questions):
                if q["id"] == current_id and i < len(valid_questions) - 1:
                    next_question = valid_questions[i + 1]
                    break
            
            # If we reached the end, loop back to the first question
            if next_question is None:
                next_question = valid_questions[0]
            
            # Update session state with the next question
            st.session_state.current_question = next_question
            st.session_state.show_answer = False
            st.session_state.user_answer = ""
            st.session_state.hint_index = 0
            
            # Clear all input fields
            clear_input_fields()
            
            # Add JavaScript to scroll to the top of the page
            st.markdown("""
            <script>
                window.scrollTo(0, 0);
            </script>
            """, unsafe_allow_html=True)
            
            st.rerun()

# ==================================
# Display Logic: Theory Questions
# ==================================
def display_theory_question(question):
    """
    Display a theory question with a friendly, encouraging interface.
    """
    try:
        st.title("🧠 Theory Round: Test Your AI Knowledge")
        st.subheader(f"Topic: {question['category']}")
        
        # Study Guide Button
        with st.expander("📚 Prepare for this question type", expanded=False):
            st.markdown("### Review study materials before attempting the question")
            st.markdown("Understanding the underlying concepts will help you solve the problem more effectively!")
            
            if st.button("📖 Review Study Guide", key=f"study_guide_btn_{question['id']}"):
                display_study_guide(question['category'], "theory")
        
        # Pre-Question Checklist
        with st.expander("✅ Pre-Question Checklist", expanded=False):
            st.markdown("### Before you answer, check these off:")
            checklist = generate_question_checklist("theory")
            for item in checklist:
                st.markdown(f"{item}")
        
        # Display the question
        st.markdown(f"### Question {question['id']}")
        st.markdown(question["question"])
        
        # Display the options
        st.markdown("### Options")
        for option in question["options"]:
            st.markdown(f"- {option}")
        
        # Hints on Demand - Added to match coding questions
        if "hints" in question:
            display_hints(question["hints"])
        
        # Fun fact related to the category
        theory_facts = {
            "Machine Learning Fundamentals": "Did you know? The term 'Machine Learning' was coined in 1959 by Arthur Samuel, an IBM pioneer!",
            "Machine Learning Algorithms": "Fun fact: A single training run for a large language model can cost millions of dollars in computing resources!",
            "Natural Language Processing": "Cool tidbit: The average human vocabulary is about 20,000-35,000 words, while GPT models are trained on billions of words!",
            "Computer Vision": "Eye-opening fact: Humans can recognize about 10,000 object categories, while modern CV systems can classify millions!",
            "Reinforcement Learning": "Game on! AlphaGo made a move in its match against Lee Sedol that had a 1 in 10,000 chance of being played by a human!",
            "Statistics": "Number nugget: The normal distribution was discovered in the 18th century while analyzing errors in astronomical measurements!",
            "Neural Networks": "Brain-inspired fact: The human brain has about 86 billion neurons, while even the largest neural networks have only billions of parameters!",
            "Deep Learning": "Deep thought: The term 'deep learning' refers to having multiple hidden layers in a neural network!",
            "Evaluation Metrics": "Measuring success: A good model isn't just accurate - it needs to be fair, robust, and explainable too!"
        }
        
        if question['category'] in theory_facts:
            st.info(theory_facts[question['category']])
        
        with st.expander("📚 Question Context (Click if you need more background)", expanded=False):
            st.markdown("""
            Theory questions test your understanding of AI/ML concepts that might come up in interviews.
            Don't worry if you don't know everything - this is a learning journey!
            
            **Quick Tips:**
            * Read each option carefully - sometimes the differences are subtle
            * Think about real-world applications of these concepts
            * If unsure, try to eliminate obviously wrong answers first
            """)
        
        st.markdown("### 🤔 Question")
        st.markdown(question["question"])
        
        st.markdown("### 🎯 Options")
        selected_option = st.radio(
            "What's your best guess?",
            [option[2:] for option in question["options"]],
            key="theory_options"
        )
        
        selected_letter = None
        for i, option in enumerate(question["options"]):
            if option[2:] == selected_option:
                selected_letter = option[0]
                break
        
        if st.button("✅ Submit Answer", key="submit_theory", use_container_width=True):
            st.session_state.user_answer = selected_letter
            st.session_state.show_answer = True
            
            st.session_state.session_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "category": "theory",
                "question_id": question["id"],
                "question_category": question["category"],
                "user_answer": selected_letter,
                "correct_answer": question["correct_answer"]
            })
            
            is_correct = selected_letter == question["correct_answer"]
            st.session_state.user_scores["theory"].append(int(is_correct))
        
        if st.session_state.show_answer:
            st.markdown("---")
            st.markdown("## 🎓 Answer Breakdown")
            feedback = evaluate_theory_answer(st.session_state.user_answer, question["correct_answer"])
            
            if feedback["is_correct"]:
                st.success(f"🎉 Woohoo! You nailed it! The answer is {question['correct_answer']}.")
            else:
                st.info(f"Nice try! The correct answer is {question['correct_answer']}. Let's learn why!")
            
            st.markdown("### 💡 Explanation")
            st.markdown(question["explanation"])
            
            # Add a motivational message
            motivational_messages = [
                "Every question you tackle makes you stronger for your interviews!",
                "Making mistakes is how we learn - that's the secret to mastery!",
                "You're building neural pathways with every question - keep it up!",
                "Great job engaging with this topic - you're on your way to interview success!",
                "Remember: even AI experts were beginners once. You're on the right path!"
            ]
            import random
            st.markdown(f"### 💪 Keep Going!")
            st.markdown(f"*{random.choice(motivational_messages)}*")
            
            if st.button("🚀 Next Question", key="next_theory", use_container_width=True):
                # Find the next question in the list
                current_id = st.session_state.current_question["id"]
                next_question = None
                
                # Get all valid theory questions
                valid_questions = []
                for q in THEORY_QUESTIONS:
                    is_valid, _ = validate_question(q, "theory")
                    if is_valid:
                        valid_questions.append(q)
                
                # Sort by ID
                valid_questions.sort(key=lambda x: x["id"])
                
                # Find the next question
                for i, q in enumerate(valid_questions):
                    if q["id"] == current_id and i < len(valid_questions) - 1:
                        next_question = valid_questions[i + 1]
                        break
                
                # If we reached the end, loop back to the first question
                if next_question is None:
                    next_question = valid_questions[0]
                
                # Update session state with the next question
                st.session_state.current_question = next_question
                st.session_state.show_answer = False
                st.session_state.user_answer = ""
                st.session_state.hint_index = 0
                
                # Clear all text input fields by regenerating their keys
                clear_input_fields()
                
                # Add JavaScript to scroll to the top of the page
                st.markdown("""
                <script>
                    window.scrollTo(0, 0);
                </script>
                """, unsafe_allow_html=True)
                
                st.rerun()
    except Exception as e:
        st.error(f"Error displaying theory question: {e}")
        st.error("This might not be a theory question. Please go back and select a different question.")

# ==================================
# Display Logic: Algorithm Design
# ==================================
def display_algorithm_design_question(question):
    """
    Display an algorithm design question with a friendly, encouraging interface.
    """
    st.title("🔮 Algorithm Design Round: Build AI Solutions")
    st.subheader(f"{question['title']} ({question['difficulty']})")
    st.markdown(f"*Category: {question['category']}*")
    
    # Study Guide Button
    with st.expander("📚 Prepare for this question type", expanded=False):
        st.markdown("### Review study materials before attempting the question")
        st.markdown("Understanding the underlying concepts will help you solve the problem more effectively!")
        
        if st.button("📖 Review Study Guide", key=f"study_guide_btn_{question['id']}"):
            display_study_guide(question['category'], "algorithm_design")
    
    # Pre-Question Checklist
    with st.expander("✅ Pre-Question Checklist", expanded=False):
        st.markdown("### Before you start designing, check these off:")
        checklist = generate_question_checklist("algorithm_design")
        for item in checklist:
            st.markdown(f"{item}")
    
    # Fun design tips
    with st.expander("🧙‍♂️ Design Wizard Tips (Click for some magic)", expanded=False):
        st.markdown("""
        ### How to Approach Algorithm Design Questions
        
        1. **Understand the problem** - Take your time to really get what's being asked
        2. **Think about the data** - What inputs do you have? What outputs do you need?
        3. **Start simple** - Begin with a basic approach, then refine it
        4. **Consider trade-offs** - Speed vs. memory, accuracy vs. complexity
        5. **Explain your thinking** - Interviewers love hearing your thought process!
        
        Remember: There's often no single "right" answer - it's about your approach and reasoning!
        """)
    
    # Problem statement with friendly formatting
    st.markdown("### 🎯 Your Challenge")
    st.markdown(question["problem"])
    
    # Hints section - Updated to use the display_hints function for consistency
    if "hints" in question:
        display_hints(question["hints"])
    
    # Remove the old hints implementation
    # st.markdown("### 💡 Helpful Hints")
    # for i, hint in enumerate(question["hints"]):
    #     if i < st.session_state.hint_index + 1:
    #         st.markdown(f"- {hint}")
    
    # # Show hint button
    # if st.session_state.hint_index < len(question["hints"]) - 1:
    #     if st.button("Show Next Hint"):
    #         st.session_state.hint_index += 1
    #         st.rerun()
    
    # Design approach section
    if "design_approach" in question:
        with st.expander("📝 Design Approach Guidelines", expanded=False):
            if "explanation" in question["design_approach"]:
                st.markdown("#### Approach Explanation")
                st.markdown(question["design_approach"]["explanation"])
            
            if "pseudocode" in question["design_approach"]:
                st.markdown("#### Pseudocode Example")
                st.code(question["design_approach"]["pseudocode"], language="python")
            
            if "key_considerations" in question["design_approach"]:
                st.markdown("#### Key Considerations")
                st.markdown(question["design_approach"]["key_considerations"])
            
            if "evaluation_criteria" in question["design_approach"]:
                st.markdown("#### Evaluation Criteria")
                st.markdown(question["design_approach"]["evaluation_criteria"])
    
    # User input with encouraging prompts
    st.markdown("### 🧩 Your Solution Approach")
    st.markdown("*Don't worry about getting everything perfect - focus on your overall strategy!*")
    
    user_solution = st.text_area(
        "Describe your approach (pseudocode, steps, or explanation):",
        height=200,
        key=f"algo_design_solution_{question['id']}"
    )
    
    # Submit solution
    if st.button("Submit Solution", key=f"submit_algo_design_{question['id']}"):
        if user_solution.strip():
            st.success("Solution submitted! Great job thinking through this problem.")
            
            # Save to session history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_entry = {
                "timestamp": timestamp,
                "category": "algorithm_design",
                "question_id": question["id"],
                "question_title": question["title"],
                "user_solution": user_solution,
                "score": None  # No automatic scoring for algorithm design
            }
            st.session_state.session_history.append(history_entry)
            
            # Show the model solution
            st.session_state.show_answer = True
        else:
            st.warning("Please enter your solution before submitting.")
    
    # Show model solution after submission
    if st.session_state.show_answer and "design_approach" in question:
        st.markdown("### 🌟 Model Solution")
        
        if "explanation" in question["design_approach"]:
            st.markdown("#### Approach")
            st.markdown(question["design_approach"]["explanation"])
        
        if "pseudocode" in question["design_approach"]:
            st.markdown("#### Pseudocode")
            st.code(question["design_approach"]["pseudocode"], language="python")
        
        # Self-evaluation section
        st.markdown("### 🔍 Self-Evaluation")
        st.markdown("Compare your solution with the model solution and rate yourself:")
        
        col1, col2 = st.columns(2)
        with col1:
            approach_rating = st.slider(
                "Approach Quality (1-5):",
                1, 5, 3,
                key=f"approach_rating_{question['id']}"
            )
        
        with col2:
            completeness_rating = st.slider(
                "Solution Completeness (1-5):",
                1, 5, 3,
                key=f"completeness_rating_{question['id']}"
            )
        
        if st.button("Save Self-Evaluation", key=f"save_eval_{question['id']}"):
            # Update the last history entry with self-evaluation
            if st.session_state.session_history:
                for entry in reversed(st.session_state.session_history):
                    if entry["question_id"] == question["id"] and entry["category"] == "algorithm_design":
                        entry["self_evaluation"] = {
                            "approach_rating": approach_rating,
                            "completeness_rating": completeness_rating,
                            "average_rating": (approach_rating + completeness_rating) / 2
                        }
                        entry["score"] = (approach_rating + completeness_rating) / 2  # Average score
                        break
                
                st.success("Self-evaluation saved!")
                
                # Update user scores
                if "algorithm_design" not in st.session_state.user_scores:
                    st.session_state.user_scores["algorithm_design"] = []
                
                st.session_state.user_scores["algorithm_design"].append({
                    "question_id": question["id"],
                    "score": (approach_rating + completeness_rating) / 2,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Save session data
                save_session_data()
                
                # Add a Next Question button
                st.markdown("---")
                if st.button("🚀 Next Question", key="next_algorithm_design", use_container_width=True):
                    # Find the next question in the list
                    current_id = st.session_state.current_question["id"]
                    next_question = None
                    
                    # Get all valid algorithm design questions
                    valid_questions = []
                    for q in ALGORITHM_DESIGN_QUESTIONS:
                        is_valid, _ = validate_question(q, "algorithm_design")
                        if is_valid:
                            valid_questions.append(q)
                    
                    # Sort by ID
                    valid_questions.sort(key=lambda x: x["id"])
                    
                    # Find the next question
                    for i, q in enumerate(valid_questions):
                        if q["id"] == current_id and i < len(valid_questions) - 1:
                            next_question = valid_questions[i + 1]
                            break
                    
                    # If we reached the end, loop back to the first question
                    if next_question is None:
                        next_question = valid_questions[0]
                    
                    # Update session state with the next question
                    st.session_state.current_question = next_question
                    st.session_state.show_answer = False
                    st.session_state.user_answer = ""
                    st.session_state.hint_index = 0
                    
                    # Clear all input fields
                    clear_input_fields()
                    
                    # Add JavaScript to scroll to the top of the page
                    st.markdown("""
                    <script>
                        window.scrollTo(0, 0);
                    </script>
                    """, unsafe_allow_html=True)
                    
                    st.rerun()

# ==================================
# Display Progress Dashboard
# ==================================
def display_dashboard_page():
    """
    Display the user's progress dashboard.
    """
    st.title("📊 Your Progress Dashboard")
    
    # Get session history
    session_manager = get_session_manager()
    session_history = session_manager.get_session_history()
    
    if not session_history:
        st.info("No practice sessions recorded yet. Start practicing to see your progress!")
        return
    
    display_dashboard(st.session_state.session_history, st.session_state.user_scores)

def display_leaderboard_page():
    """
    Display the leaderboard page.
    """
    st.title("🏆 Leaderboard: Top Performers")
    display_leaderboard()

def display_leaderboard():
    """
    Display the leaderboard with top performers.
    """
    st.subheader("Leaderboard (Demo)")
    
    if 'leaderboard' not in st.session_state or len(st.session_state.leaderboard) == 0:
        st.info("No leaderboard data yet. Complete some questions to appear on the leaderboard!")
        return
    
    # Sort the leaderboard by score (descending)
    sorted_lb = sorted(st.session_state.leaderboard, key=lambda x: x.get("score", 0), reverse=True)
    
    # Create a DataFrame for display
    lb_df = pd.DataFrame(sorted_lb)
    
    # Format the DataFrame
    if not lb_df.empty:
        lb_df.columns = [col.capitalize() for col in lb_df.columns]
        if "Score" in lb_df.columns:
            lb_df["Score"] = lb_df["Score"].apply(lambda x: f"{x:.1f}%")
        if "Efficiency" in lb_df.columns:
            lb_df["Efficiency"] = lb_df["Efficiency"].apply(lambda x: f"{x:.1f}%")
    
    # Display the leaderboard
    st.dataframe(lb_df, use_container_width=True)
    
    # Add a fun message
    if not lb_df.empty:
        top_user = lb_df.iloc[0]["User"] if "User" in lb_df.columns else "Anonymous"
        st.markdown(f"🎉 Congratulations to **{top_user}** for leading the pack!")
    
    # Leaderboard explanation
    with st.expander("How the leaderboard works"):
        st.markdown("""
        ### Leaderboard Scoring
        
        The leaderboard ranks users based on their performance in interview questions:
        
        - **Score**: Overall performance score (70% correctness, 30% efficiency)
        - **Efficiency**: How optimal your solutions are
        
        Keep practicing to improve your ranking! Your best scores are always kept.
        """)

def display_progress_dashboard():
    tab1, tab2, tab3, tab4 = st.tabs(["Progress & Performance", "Recommendations", "Session Management", "Leaderboard"])
    
    with tab1:
        display_dashboard(st.session_state.session_history, st.session_state.user_scores)
    
    with tab2:
        display_recommendations(st.session_state.session_history)
    
    with tab3:
        st.subheader("Session Management")
        session_manager = get_session_manager()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Previous Sessions")
            sessions = session_manager.get_all_sessions(st.session_state.user_id)
            if sessions:
                sessions_df = pd.DataFrame(sessions)
                sessions_df.columns = ["Session ID", "Timestamp"]
                sessions_df = sessions_df.set_index("Session ID")
                
                st.dataframe(sessions_df)
                
                selected_session = st.selectbox(
                    "Select a session to load:",
                    [s["id"] for s in sessions],
                    format_func=lambda x: f"Session {x} - {next((s['timestamp'] for s in sessions if s['id'] == x), '')}"
                )
                
                if st.button("Load Selected Session"):
                    if load_session_data(selected_session):
                        st.success(f"Session {selected_session} loaded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to load session data.")
            else:
                st.info("No previous sessions found.")
        
        with col2:
            st.markdown("### Export Options")
            
            if st.button("Export Session to JSON"):
                if st.session_state.db_session_id:
                    file_path = export_session_to_json(st.session_state.db_session_id)
                    if file_path:
                        st.success(f"Session exported to {file_path}")
                    else:
                        st.error("Failed to export session.")
                else:
                    success, session_id = save_session_data()
                    if success:
                        file_path = export_session_to_json(session_id)
                        if file_path:
                            st.success(f"Session exported to {file_path}")
                        else:
                            st.error("Failed to export session.")
                    else:
                        st.error("Failed to save session data for export.")
            
            if st.button("Export Performance to CSV"):
                file_path = export_performance_to_csv(st.session_state.user_id)
                if file_path:
                    st.success(f"Performance data exported to {file_path}")
                else:
                    st.error("Failed to export performance data.")
            
            if st.button("Backup Database"):
                file_path = backup_database()
                if file_path:
                    st.success(f"Database backed up to {file_path}")
                else:
                    st.error("Failed to backup database.")
            
            st.markdown("### User Settings")
            new_user_id = st.text_input("User ID:", value=st.session_state.user_id)
            if st.button("Update User ID"):
                st.session_state.user_id = new_user_id
                st.success(f"User ID updated to {new_user_id}")
    
    with tab4:
        display_leaderboard_page()

# ===============================
# Question Selection Logic
# ===============================
def handle_question_selection(questions, category_label):
    """
    Handle the selection and display of questions.
    """
    # Add JavaScript to scroll to the top of the page
    st.markdown("""
    <script>
        // Function to scroll to the top of the page
        function scrollToTop() {
            window.scrollTo(0, 0);
        }
        
        // Execute the function
        scrollToTop();
    </script>
    """, unsafe_allow_html=True)
    
    # Create lists for selection
    question_titles = []
    valid_questions = []
    question_map = {}
    
    for q in questions:
        # Validate the question before adding it to the selection list
        is_valid, error_msg = validate_question(q, st.session_state.current_category)
        if not is_valid:
            continue  # Skip invalid questions
            
        valid_questions.append(q)
        
        if st.session_state.current_category == "theory":
            # Theory questions have different structure
            title = f"{q['id']}. {q['category']} - {q['question'][:50]}..."
        else:
            # Coding and Algorithm Design questions
            title = f"{q['id']}. {q['title']} ({q['difficulty']})"
        
        question_titles.append(title)
        question_map[title] = q
    
    if not question_titles:
        st.error(f"No valid questions found for {category_label}. Please check your question data files.")
        return
    
    # Create a unique key for the selectbox
    selectbox_key = f"question_selection_{st.session_state.current_category}"
    
    # Initialize selected_question_title in session state if not present
    if f"selected_question_{st.session_state.current_category}" not in st.session_state:
        st.session_state[f"selected_question_{st.session_state.current_category}"] = question_titles[0]
    
    if st.session_state.current_question is None:
        if st.session_state.current_category == "coding":
            st.title("🧩 Coding Round: Algorithmic Problem Solving")
        elif st.session_state.current_category == "theory":
            st.title("🧠 Theory Round: Test Your AI Knowledge")
        elif st.session_state.current_category == "algorithm_design":
            st.title("🔮 Algorithm Design Round: Build AI Solutions")
        else:
            st.header(category_label)
        
        # Debug information for theory questions
        if st.session_state.current_category == "theory":
            with st.expander("Debug Information (Admin Only)", expanded=False):
                for i, q in enumerate(questions):
                    is_valid, error_msg = validate_question(q, st.session_state.current_category)
                    if not is_valid:
                        st.error(f"Question {q.get('id', i)}: {error_msg}")
        
        # Add question selection to sidebar immediately
        st.sidebar.markdown("### Select a Question")
        
        selected_question = st.sidebar.selectbox(
            "Choose a question to practice:",
            question_titles,
            index=question_titles.index(st.session_state[f"selected_question_{st.session_state.current_category}"]),
            key=selectbox_key
        )
        
        # Store the selected question title in session state
        st.session_state[f"selected_question_{st.session_state.current_category}"] = selected_question
        
        # Display a welcome message and instructions
        st.markdown(f"### Welcome to the {category_label} section!")
        st.markdown("Click 'Start Practice' to begin with the selected question.")
        
        # Add a button to start practicing
        if st.button("Start Practice", key=f"start_practice_{st.session_state.current_category}"):
            # Set the current question based on selection
            st.session_state.current_question = question_map[selected_question]
            
            # Reset answer state when starting a new question
            st.session_state.show_answer = False
            st.session_state.user_answer = ""
            st.session_state.hint_index = 0
            st.rerun()
    else:
        try:
            # Validate the current question before displaying it
            is_valid, error_msg = validate_question(st.session_state.current_question, st.session_state.current_category)
            if not is_valid:
                st.error(f"Current question is invalid: {error_msg}")
                st.error("Please go back and select a different question.")
                
                # Add a button to go back to question selection
                if st.button("Go Back to Question Selection"):
                    st.session_state.current_question = None
                    st.session_state.show_answer = False
                    st.session_state.user_answer = ""
                    st.session_state.hint_index = 0
                    st.rerun()
                return
            
            # Add a selection box to change questions directly
            if st.session_state.current_category in ["coding", "algorithm_design", "theory"]:
                st.sidebar.markdown("### Change Question")
                
                # Find the current question in the list
                current_title = None
                for title, q in question_map.items():
                    if q["id"] == st.session_state.current_question["id"]:
                        current_title = title
                        break
                
                # Create a unique key for the selectbox
                change_question_key = f"change_question_{st.session_state.current_category}"
                
                # Show the selection box with the current question selected
                if current_title and question_titles:
                    new_question_title = st.sidebar.selectbox(
                        "Select a different question:",
                        question_titles,
                        index=question_titles.index(current_title) if current_title in question_titles else 0,
                        key=change_question_key
                    )
                    
                    # If the user selects a different question, update the session state
                    if new_question_title != current_title:
                        st.session_state.current_question = question_map[new_question_title]
                        st.session_state.show_answer = False
                        st.session_state.user_answer = ""
                        st.session_state.hint_index = 0
                        st.rerun()
            
            if st.session_state.current_category == "coding":
                display_coding_question(st.session_state.current_question)
            elif st.session_state.current_category == "theory":
                display_theory_question(st.session_state.current_question)
            elif st.session_state.current_category == "algorithm_design":
                display_algorithm_design_question(st.session_state.current_question)
        except Exception as e:
            st.error(f"Error displaying question: {e}")
            st.error("Please go back and select a different question.")
            
            # Add a button to go back to question selection
            if st.button("Go Back to Question Selection"):
                st.session_state.current_question = None
                st.session_state.show_answer = False
                st.session_state.user_answer = ""
                st.session_state.hint_index = 0
                st.rerun()
        
        # Use a unique key for the back button based on category and question ID
        back_button_key = f"back_to_selection_{st.session_state.current_category}_{st.session_state.current_question.get('id', 0)}"
        if st.sidebar.button("Back to Question Selection", key=back_button_key):
            st.session_state.current_question = None
            st.session_state.show_answer = False
            st.session_state.user_answer = ""
            st.session_state.hint_index = 0
            st.rerun()

# ===============================
# Main App Entry
# ===============================
def display_welcome():
    """
    Display the welcome message and instructions.
    """
    st.title("🚀 Welcome to AI Interview Mastery Trainer")
    st.markdown("""
    ### Hey there, future AI superstar! 👋
    
    Welcome to your fun, friendly training ground for acing those AI Engineer interviews!
    
    Here's what we've got in store for you:
    
    🧩 **Coding Round**: Tackle algorithm puzzles without breaking a sweat
    
    🧠 **Theory Round**: Master AI/ML concepts with a smile
    
    🔮 **Algorithm Design**: Create predictive magic like a pro
    
    Pick a category on the left and let's get started! Remember, practice makes perfect (and we're here to make practice fun)!
    """)

def clear_input_fields():
    """
    Clear all input fields from the session state to ensure a fresh start for each question.
    """
    # List of prefixes to clear
    prefixes_to_clear = [
        "brute_force_",
        "optimal_solution_",
        "algo_design_solution_",
        "theory_answer_",
        "approach_rating_",
        "completeness_rating_",
        "submit_",
        "save_eval_"
    ]
    
    # Clear all keys that start with any of the prefixes
    for key in list(st.session_state.keys()):
        for prefix in prefixes_to_clear:
            if key.startswith(prefix):
                del st.session_state[key]
                break

def main():
    st.sidebar.title("✨ Your Journey")
    category = st.sidebar.radio(
        "What would you like to practice today?",
        ["Home", "Dashboard", "Coding", "Theory", "Algorithm Design", "Study Progress"]
    )
    
    # Initialize session state if not already done
    initialize_session_state()
    
    # Check if we need to navigate to a study guide
    if 'navigate_to_study_guide' in st.session_state and st.session_state.navigate_to_study_guide:
        guide_info = st.session_state.navigate_to_study_guide
        # Display the study guide
        st.title(f"Study Guide: {guide_info['topic']}")
        display_study_guide(guide_info['topic'], guide_info['category'])
        
        # Add a button to return to study progress
        if st.button("← Back to Study Progress"):
            st.session_state.navigate_to_study_guide = None
            st.rerun()
            
        # Early return to prevent showing other content
        return
    
    # Reset current question when changing categories
    if "previous_category" not in st.session_state:
        st.session_state.previous_category = None
        
    if st.session_state.previous_category != category:
        st.session_state.current_question = None
        st.session_state.show_answer = False
        st.session_state.user_answer = ""
        st.session_state.hint_index = 0
        st.session_state.previous_category = category
    
    if category == "Home":
        display_welcome()
    elif category == "Dashboard":
        display_progress_dashboard()
    elif category == "Coding":
        st.session_state.current_category = "coding"
        handle_question_selection(CODING_QUESTIONS, "Coding Questions")
    elif category == "Theory":
        st.session_state.current_category = "theory"
        handle_question_selection(THEORY_QUESTIONS, "Theory Questions")
    elif category == "Algorithm Design":
        st.session_state.current_category = "algorithm_design"
        handle_question_selection(ALGORITHM_DESIGN_QUESTIONS, "Algorithm Design Questions")
    elif category == "Study Progress":
        st.session_state.current_category = "study_progress"
        display_study_progress()
    else:
        st.error(f"Unknown page: {st.session_state.current_page}")

# ===============================
# Run the App
# ===============================
if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            get_session_manager().close()
        except:
            pass

def display_study_progress():
    st.title("📊 Study Progress")
    st.markdown("### Your Study Progress")
    st.write("This is your study progress page.")
