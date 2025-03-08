"""
Study guides for the AI Interview Mastery Trainer.

This module contains structured learning materials for various interview topics,
including coding concepts, theory questions, and algorithm design approaches.

Author: Nicole LeGuern (CodeQueenie)
"""

STUDY_GUIDES = {
    "coding": {
        "Arrays & Hashing": {
            "summary": "Arrays and hash maps (dictionaries) are fundamental data structures that allow for efficient data storage and retrieval.",
            "key_points": [
                "Arrays provide O(1) access by index, but O(n) search for values",
                "Hash maps offer O(1) average time complexity for lookups, insertions, and deletions",
                "Hash collisions can degrade performance to O(n) in worst case",
                "Python dictionaries are implemented as hash tables"
            ],
            "use_cases": ["Two Sum", "Group Anagrams", "Valid Anagram", "Contains Duplicate"],
            "example": """
# Using a hash map to solve the Two Sum problem
def two_sum(nums, target):
    # Value -> Index mapping
    num_map = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    
    return []  # No solution found
            """,
            "fun_fact": "The concept of hashing dates back to the 1950s, but modern hash functions like SHA-256 can process billions of operations per second!"
        },
        "Two Pointers": {
            "summary": "Two pointers is a technique where two references are used to traverse an array or list, often moving toward each other or at different speeds.",
            "key_points": [
                "Useful for problems involving sorted arrays",
                "Can reduce time complexity from O(n²) to O(n)",
                "Common patterns: start/end pointers, fast/slow pointers",
                "Often used with sliding window technique"
            ],
            "use_cases": ["Valid Palindrome", "Container With Most Water", "3Sum", "Remove Duplicates"],
            "example": """
# Using two pointers to check if a string is a palindrome
def is_palindrome(s):
    # Clean the string: remove non-alphanumeric and convert to lowercase
    s = ''.join(c.lower() for c in s if c.isalnum())
    
    # Use two pointers: one from start, one from end
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    
    return True
            """,
            "fun_fact": "The 'tortoise and hare' algorithm (a fast/slow pointer technique) can detect cycles in linked lists and was invented by Robert W. Floyd in the 1960s!"
        },
        "Dynamic Programming": {
            "summary": "Dynamic Programming (DP) optimizes recursive problems by storing solutions to subproblems to avoid redundant calculations.",
            "key_points": [
                "Two key properties: overlapping subproblems and optimal substructure",
                "Two approaches: top-down (memoization) and bottom-up (tabulation)",
                "Can transform exponential time complexity to polynomial",
                "Common in optimization problems (maximizing/minimizing)"
            ],
            "use_cases": ["Fibonacci Sequence", "Longest Common Subsequence", "Knapsack Problem", "Coin Change"],
            "example": """
# Fibonacci using memoization (top-down DP)
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

# Fibonacci using tabulation (bottom-up DP)
def fibonacci_tabulation(n):
    if n <= 1:
        return n
        
    # Initialize table
    dp = [0] * (n + 1)
    dp[1] = 1
    
    # Fill the table
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
        
    return dp[n]
            """,
            "fun_fact": "The term 'Dynamic Programming' was coined by Richard Bellman in the 1950s. He chose the word 'dynamic' to capture the time-varying aspect of the problems, and 'programming' referred to planning and decision-making!"
        },
        "Sliding Window": {
            "summary": "The sliding window technique involves maintaining a 'window' of elements and sliding it through an array or string to solve problems efficiently.",
            "key_points": [
                "Reduces time complexity from O(n²) to O(n) for many problems",
                "Two types: fixed-size window and variable-size window",
                "Often used with arrays and strings",
                "Useful for finding subarrays/substrings with specific properties"
            ],
            "use_cases": ["Maximum Sum Subarray of Size K", "Longest Substring Without Repeating Characters", "Minimum Size Subarray Sum"],
            "example": """
# Find maximum sum subarray of size k
def max_sum_subarray(arr, k):
    n = len(arr)
    if n < k:
        return None
    
    # Compute sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide the window and update max_sum
    for i in range(n - k):
        window_sum = window_sum - arr[i] + arr[i + k]
        max_sum = max(max_sum, window_sum)
    
    return max_sum
            """,
            "fun_fact": "The sliding window technique is inspired by the concept of a physical window that 'slides' along a wall to provide different views. In algorithms, it gives us different 'views' of the data!"
        },
        "Binary Search": {
            "summary": "Binary search is a divide-and-conquer algorithm that finds the position of a target value in a sorted array by repeatedly dividing the search space in half.",
            "key_points": [
                "Requires a sorted array",
                "Time complexity: O(log n)",
                "Space complexity: O(1) for iterative implementation",
                "Can be applied to more complex problems beyond simple search"
            ],
            "use_cases": ["Search in Sorted Array", "Find First and Last Position", "Search in Rotated Sorted Array"],
            "example": """
# Binary search implementation
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid integer overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found
            """,
            "fun_fact": "Binary search was first published in 1946, but the first bug-free implementation wasn't published until 1962! Even experienced programmers often get the details wrong."
        }
    },
    "theory": {
        "Data Structures": {
            "summary": "Data structures are specialized formats for organizing, processing, retrieving, and storing data to suit specific purposes.",
            "key_points": [
                "Choosing the right data structure can dramatically impact performance",
                "Common structures: arrays, linked lists, stacks, queues, trees, graphs, hash tables",
                "Each structure has trade-offs in terms of time and space complexity",
                "Data structures can be primitive, composite, abstract, or concrete"
            ],
            "use_cases": ["Database indexing", "Network routing", "Compiler design", "Operating systems"],
            "example": """
# Implementing a stack using a list in Python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
            """,
            "fun_fact": "The B-tree data structure, commonly used in databases and file systems, was invented by Rudolf Bayer and Edward M. McCreight while working at Boeing Research Labs in 1971!"
        },
        "Machine Learning Fundamentals": {
            "summary": "Machine learning is a field of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
            "key_points": [
                "Core types: supervised learning, unsupervised learning, reinforcement learning, and semi-supervised learning",
                "Supervised learning uses labeled data to train models that can make predictions",
                "Unsupervised learning finds patterns in unlabeled data without predefined outputs",
                "The machine learning workflow includes data collection, preprocessing, feature engineering, model selection, training, evaluation, and deployment"
            ],
            "use_cases": ["Image recognition", "Natural language processing", "Recommendation systems", "Fraud detection"],
            "example": """
# Basic supervised learning workflow with scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Data preparation
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 2. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model training
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# 4. Prediction and evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")
            """,
            "fun_fact": "The term 'machine learning' was coined by Arthur Samuel in 1959 while at IBM, where he developed a checkers-playing program that could learn from its own experience!"
        },
        "Supervised vs. Unsupervised Learning": {
            "summary": "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data.",
            "key_points": [
                "Supervised learning: Classification (discrete outputs) and Regression (continuous outputs)",
                "Unsupervised learning: Clustering, Dimensionality Reduction, Association",
                "Semi-supervised learning combines both approaches with partially labeled data",
                "Reinforcement learning uses rewards/penalties to train agents through interaction"
            ],
            "use_cases": ["Image classification", "Spam detection", "Customer segmentation", "Anomaly detection"],
            "example": """
# Supervised learning example with scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a random forest classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
            """,
            "fun_fact": "The term 'machine learning' gained popularity in the 2010s, but the concept dates back to the early days of statistics. Even Florence Nightingale used feature engineering when analyzing mortality data during the Crimean War in the 1850s!"
        },
        "Neural Networks": {
            "summary": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes (neurons) that process and transmit signals.",
            "key_points": [
                "Basic components: input layer, hidden layers, output layer, weights, biases, activation functions",
                "Training involves forward propagation and backpropagation to adjust weights",
                "Deep learning uses neural networks with many hidden layers",
                "Specialized architectures: CNNs for images, RNNs/LSTMs for sequences, Transformers for NLP"
            ],
            "use_cases": ["Image recognition", "Natural language processing", "Speech recognition", "Game playing"],
            "example": """
# Simple neural network with TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a sequential model
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
            """,
            "fun_fact": "The perceptron, the simplest form of neural network, was invented in 1957 by Frank Rosenblatt. Initial excitement was dampened when it was proven that perceptrons couldn't solve the XOR problem, leading to the first 'AI winter'!"
        }
    },
    "algorithm_design": {
        "Algorithm Design": {
            "summary": "Algorithm design is the process of creating step-by-step procedures for solving computational problems, focusing on efficiency, correctness, and scalability.",
            "key_points": [
                "Understand the problem requirements and constraints thoroughly",
                "Consider time and space complexity trade-offs",
                "Break down complex problems into smaller, manageable subproblems",
                "Validate your solution with test cases and edge cases",
                "Optimize iteratively, starting with a working solution before refining"
            ],
            "use_cases": ["Feature selection", "Time-series forecasting", "Reinforcement learning", "Recommendation systems"],
            "example": """
# Example of a feature selection algorithm using mutual information
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import numpy as np

def select_best_features(X, y, k=10):
    # Initialize the selector with mutual information
    selector = SelectKBest(mutual_info_classif, k=k)
    
    # Fit the selector to the data
    X_new = selector.fit_transform(X, y)
    
    # Get the indices of the selected features
    selected_indices = np.where(selector.get_support())[0]
    
    # Get the scores of the selected features
    scores = selector.scores_[selected_indices]
    
    # Return the selected features and their scores
    return X_new, selected_indices, scores

# Usage
X_selected, selected_features, feature_scores = select_best_features(X, y, k=5)
print(f"Selected features: {selected_features}")
print(f"Feature scores: {feature_scores}")
            """,
            "fun_fact": "The term 'algorithm' comes from the name of the 9th-century Persian mathematician Al-Khwarizmi, whose works introduced algorithmic solutions to mathematical problems in the Western world!"
        },
        "Feature Selection": {
            "summary": "Feature selection is the process of identifying and selecting the most relevant variables for a predictive model, reducing dimensionality and improving performance.",
            "key_points": [
                "Filter methods: statistical tests like correlation, chi-square, and mutual information",
                "Wrapper methods: recursive feature elimination, forward/backward selection",
                "Embedded methods: LASSO, Ridge regression, and tree-based importance",
                "Dimensionality reduction: PCA, t-SNE, and autoencoders",
                "Handling multicollinearity with variance inflation factor (VIF)"
            ],
            "use_cases": ["Predictive modeling", "Biomarker discovery", "Text classification", "Image recognition"],
            "example": """
# Comprehensive feature selection pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def feature_selection_pipeline(X, y):
    # Create a pipeline with scaling and feature selection
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selector', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold='median'
        ))
    ])
    
    # Fit the pipeline to the data
    pipeline.fit(X, y)
    
    # Get the selected feature mask
    feature_mask = pipeline.named_steps['feature_selector'].get_support()
    
    # Return the transformed data and the feature mask
    return pipeline.transform(X), feature_mask

# Usage
X_selected, selected_features_mask = feature_selection_pipeline(X, y)
selected_feature_names = X.columns[selected_features_mask].tolist()
print(f"Selected features: {selected_feature_names}")
            """,
            "fun_fact": "The 'curse of dimensionality' term was coined by Richard Bellman in 1957 to describe how the number of samples needed for statistical significance grows exponentially with the number of dimensions!"
        },
        "Time-Series Forecasting": {
            "summary": "Time-series forecasting involves analyzing historical time-ordered data to predict future values, accounting for trends, seasonality, and other temporal patterns.",
            "key_points": [
                "Statistical methods: ARIMA, SARIMA, exponential smoothing",
                "Machine learning approaches: Prophet, XGBoost with lag features",
                "Deep learning models: LSTM, GRU, and Transformer networks",
                "Feature engineering: rolling statistics, lag features, and calendar variables",
                "Evaluation metrics: RMSE, MAE, MAPE, and time-series cross-validation"
            ],
            "use_cases": ["Sales prediction", "Stock price forecasting", "Demand planning", "Anomaly detection"],
            "example": """
# SARIMA model for time series forecasting
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def build_sarima_model(time_series, order=(1,1,1), seasonal_order=(1,1,1,12)):
    # Fit the SARIMA model
    model = SARIMAX(
        time_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    
    return results

def forecast_and_evaluate(model, time_series, forecast_steps=30):
    # Generate forecast
    forecast = model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    
    # If we have actual values for the forecast period
    if len(time_series) > len(time_series) - forecast_steps:
        actual = time_series[-forecast_steps:]
        
        # Calculate error metrics
        mae = mean_absolute_error(actual, forecast_mean)
        rmse = np.sqrt(mean_squared_error(actual, forecast_mean))
        mape = np.mean(np.abs((actual - forecast_mean) / actual)) * 100
        
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
    
    return forecast_mean

# Usage
model = build_sarima_model(sales_data, order=(2,1,2), seasonal_order=(1,1,1,12))
predictions = forecast_and_evaluate(model, sales_data, forecast_steps=30)
            """,
            "fun_fact": "The Box-Jenkins methodology, which forms the foundation of ARIMA models, was developed in the 1970s by statisticians George Box and Gwilym Jenkins, revolutionizing time series analysis!"
        },
        "Reinforcement Learning": {
            "summary": "Reinforcement learning is a machine learning paradigm where an agent learns to make decisions by taking actions in an environment to maximize cumulative rewards.",
            "key_points": [
                "Key components: agent, environment, state, action, reward, policy",
                "Value-based methods: Q-learning, Deep Q-Networks (DQN)",
                "Policy-based methods: REINFORCE, Proximal Policy Optimization (PPO)",
                "Actor-Critic methods: A2C, A3C, and DDPG",
                "Exploration vs. exploitation: epsilon-greedy, UCB, Thompson sampling"
            ],
            "use_cases": ["Game AI", "Robotics", "Recommendation systems", "Resource management"],
            "example": """
# Q-learning implementation for a simple environment
import numpy as np

class QLearningAgent:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((states, actions))
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.actions = actions
    
    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
    
    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
    
    def train(self, env, episodes=1000):
        for episode in range(episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
            
            # Decay epsilon over time
            self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return self.q_table

# Usage
agent = QLearningAgent(states=100, actions=4)
q_table = agent.train(environment, episodes=1000)
            """,
            "fun_fact": "AlphaGo, which defeated world champion Lee Sedol in 2016, used a combination of reinforcement learning and Monte Carlo tree search to make moves that had a 1 in 10,000 chance of being played by a human expert!"
        }
    }
}

if __name__ == "__main__":
    for category, topics in STUDY_GUIDES.items():
        print(f"\nCategory: {category.capitalize()}")
        for topic, details in topics.items():
            print(f"\n{topic}: {details['summary']}")
