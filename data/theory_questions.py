"""
This module contains AI/ML theory questions for the AI Interview Mastery Trainer.
Each question includes:
- A question statement
- Multiple choice options
- The correct answer
- A detailed explanation

Author: Nicole LeGuern (CodeQueenie)
"""

THEORY_QUESTIONS = [
    {
        "id": 1,
        "category": "Machine Learning Fundamentals",
        "question": "What is the main difference between supervised and unsupervised learning?",
        "options": [
            "A. Supervised learning requires more computational power than unsupervised learning",
            "B. Supervised learning uses labeled data for training, while unsupervised learning uses unlabeled data",
            "C. Supervised learning is used for classification, while unsupervised learning is used for regression",
            "D. Supervised learning works with continuous variables, while unsupervised learning works with categorical variables"
        ],
        "correct_answer": "B",
        "explanation": """
The main difference between supervised and unsupervised learning is the nature of the training data:

- Supervised learning uses labeled data, where each training example has an input and a corresponding target output. The algorithm learns to map inputs to outputs based on these examples. Common supervised learning tasks include classification and regression.

- Unsupervised learning uses unlabeled data, where the algorithm must find patterns or structure in the data without explicit guidance. Common unsupervised learning tasks include clustering, dimensionality reduction, and anomaly detection.

Option A is incorrect because the computational requirements depend on the specific algorithms and datasets, not on whether the learning is supervised or unsupervised.

Option C is incorrect because both classification (predicting categorical outputs) and regression (predicting continuous outputs) are types of supervised learning.

Option D is incorrect because both supervised and unsupervised learning can work with both continuous and categorical variables.
        """
    },
    {
        "id": 2,
        "category": "Machine Learning Algorithms",
        "question": "Which algorithm is most suitable for binary classification problems?",
        "options": [
            "A. Linear Regression",
            "B. Logistic Regression",
            "C. Principal Component Analysis (PCA)",
            "D. K-means Clustering"
        ],
        "correct_answer": "B",
        "explanation": """
Logistic Regression is most suitable for binary classification problems.

- Logistic Regression is a supervised learning algorithm specifically designed for binary classification. It models the probability that an instance belongs to a particular class using the logistic function (sigmoid), which maps any real-valued number to a value between 0 and 1.

- Linear Regression (option A) is used for predicting continuous values, not for classification. While it can be adapted for binary classification by using a threshold, it's not optimal because it doesn't constrain predictions to be between 0 and 1.

- Principal Component Analysis (PCA) (option C) is a dimensionality reduction technique, not a classification algorithm. It's used to reduce the number of features while preserving as much variance as possible.

- K-means Clustering (option D) is an unsupervised learning algorithm used for clustering similar data points together. It doesn't use labeled data and isn't designed for classification tasks.
        """
    },
    {
        "id": 3,
        "category": "Neural Networks",
        "question": "What is the purpose of an activation function in a neural network?",
        "options": [
            "A. To initialize the weights of the network",
            "B. To calculate the loss of the network",
            "C. To introduce non-linearity into the network's output",
            "D. To normalize the input data"
        ],
        "correct_answer": "C",
        "explanation": """
The purpose of an activation function in a neural network is to introduce non-linearity into the network's output.

- Without activation functions, a neural network would only be able to learn linear relationships between inputs and outputs, regardless of how many layers it has. This is because a composition of linear functions is still a linear function.

- Activation functions like ReLU, sigmoid, and tanh introduce non-linearity, allowing neural networks to learn complex patterns and relationships in the data. This enables them to approximate any function, making them universal function approximators.

- Option A (initializing weights) is incorrect. Weights are typically initialized using methods like Xavier/Glorot initialization or He initialization, not by activation functions.

- Option B (calculating loss) is incorrect. Loss functions like mean squared error or cross-entropy are used to calculate the difference between predicted and actual outputs.

- Option D (normalizing input data) is incorrect. Data normalization is typically done as a preprocessing step before feeding the data into the network, not by activation functions.
        """
    },
    {
        "id": 4,
        "category": "Evaluation Metrics",
        "question": "For a highly imbalanced dataset where positive cases are rare, which evaluation metric is most appropriate?",
        "options": [
            "A. Accuracy",
            "B. Precision",
            "C. Recall",
            "D. Area Under the Precision-Recall Curve (AUPRC)"
        ],
        "correct_answer": "D",
        "explanation": """
For a highly imbalanced dataset where positive cases are rare, the Area Under the Precision-Recall Curve (AUPRC) is most appropriate.

- Accuracy (option A) can be misleading for imbalanced datasets. For example, if only 1% of cases are positive, a model that always predicts "negative" would achieve 99% accuracy without actually learning anything useful.

- Precision (option B) measures the proportion of positive predictions that are actually correct. While useful, it doesn't account for false negatives (positive cases that the model missed).

- Recall (option C) measures the proportion of actual positive cases that were correctly identified. While useful, it doesn't account for false positives.

- Area Under the Precision-Recall Curve (AUPRC) (option D) considers both precision and recall across different threshold values. It provides a more comprehensive evaluation for imbalanced datasets by focusing on the performance for the minority class. Unlike the ROC curve, which can be overly optimistic for imbalanced datasets, the PR curve is more sensitive to improvements in the rare positive class.
        """
    },
    {
        "id": 5,
        "category": "Deep Learning",
        "question": "What is the vanishing gradient problem in deep neural networks?",
        "options": [
            "A. When gradients become too large during backpropagation, causing weights to explode",
            "B. When gradients become extremely small during backpropagation, making it difficult for early layers to learn",
            "C. When the learning rate is too small, causing the network to converge slowly",
            "D. When the network has too many parameters, leading to overfitting"
        ],
        "correct_answer": "B",
        "explanation": """
The vanishing gradient problem occurs when gradients become extremely small during backpropagation, making it difficult for early layers to learn.

- During backpropagation, gradients are propagated backward through the network to update the weights. With certain activation functions (like sigmoid or tanh) and deep architectures, these gradients can become exponentially small as they are propagated to earlier layers.

- As a result, the weights in the early layers of the network update very slowly or not at all, effectively preventing these layers from learning. This was a major obstacle in training deep networks before the introduction of techniques to address this issue.

- Option A describes the exploding gradient problem, which is the opposite of the vanishing gradient problem.

- Option C describes a problem with the learning rate, not with gradients specifically.

- Option D describes overfitting, which is unrelated to the gradient flow during training.

Solutions to the vanishing gradient problem include:
1. Using activation functions like ReLU that don't saturate for positive inputs
2. Implementing architectures with skip connections (like ResNets)
3. Using batch normalization
4. Employing proper weight initialization techniques
        """
    },
    {
        "id": 6,
        "category": "Natural Language Processing",
        "question": "What is the main advantage of transformer models like BERT over traditional RNN-based models for NLP tasks?",
        "options": [
            "A. Transformers require less training data than RNNs",
            "B. Transformers can process input sequences in parallel, while RNNs process them sequentially",
            "C. Transformers are smaller and more computationally efficient than RNNs",
            "D. Transformers can only be used for text data, while RNNs can be used for any sequential data"
        ],
        "correct_answer": "B",
        "explanation": """
The main advantage of transformer models like BERT over traditional RNN-based models is that transformers can process input sequences in parallel, while RNNs process them sequentially.

- RNNs (including LSTMs and GRUs) process sequences one element at a time, with each step depending on the output of the previous step. This sequential nature makes them difficult to parallelize and slow to train on long sequences.

- Transformers, introduced in the "Attention is All You Need" paper, replace recurrence with self-attention mechanisms. This allows them to process all elements of a sequence simultaneously (in parallel), significantly speeding up training.

- Additionally, transformers can capture long-range dependencies in text more effectively than RNNs, which often struggle with long sequences due to the vanishing gradient problem.

- Option A is incorrect because transformers typically require more training data than RNNs, not less.

- Option C is incorrect because transformers are generally larger and more computationally intensive than RNNs, not smaller or more efficient.

- Option D is incorrect because both transformers and RNNs can be applied to any type of sequential data, not just text.
        """
    },
    {
        "id": 7,
        "category": "Machine Learning Concepts",
        "question": "What is the bias-variance tradeoff in machine learning?",
        "options": [
            "A. The tradeoff between model complexity and training time",
            "B. The tradeoff between the number of features and model accuracy",
            "C. The tradeoff between underfitting (high bias) and overfitting (high variance)",
            "D. The tradeoff between the size of the training set and the model's generalization ability"
        ],
        "correct_answer": "C",
        "explanation": """
The bias-variance tradeoff refers to the tradeoff between underfitting (high bias) and overfitting (high variance) in machine learning models.

- Bias refers to the error introduced by approximating a real-world problem with a simplified model. High bias can lead to underfitting, where the model fails to capture the underlying pattern in the data.

- Variance refers to the model's sensitivity to small fluctuations in the training data. High variance can lead to overfitting, where the model learns the noise in the training data rather than the underlying pattern.

- The tradeoff occurs because as we increase model complexity to reduce bias (make the model fit the training data better), we typically increase variance (make the model more sensitive to variations in the training data).

- Finding the right balance between bias and variance is crucial for creating models that generalize well to unseen data.

- Option A is incorrect because while more complex models may take longer to train, this is not the bias-variance tradeoff.

- Option B is incorrect because the number of features is just one aspect of model complexity.

- Option D is incorrect because while the size of the training set affects the model's generalization ability, this is not the bias-variance tradeoff.
        """
    },
    {
        "id": 8,
        "category": "Computer Vision",
        "question": "What is the purpose of pooling layers in Convolutional Neural Networks (CNNs)?",
        "options": [
            "A. To introduce non-linearity into the network",
            "B. To reduce spatial dimensions and extract dominant features",
            "C. To normalize the activations of the previous layer",
            "D. To connect all neurons from one layer to all neurons in the next layer"
        ],
        "correct_answer": "B",
        "explanation": """
The purpose of pooling layers in Convolutional Neural Networks (CNNs) is to reduce spatial dimensions and extract dominant features.

- Pooling layers (like max pooling or average pooling) downsample the feature maps produced by convolutional layers. This reduces the spatial dimensions (width and height) of the feature maps, which has several benefits:

  1. It reduces the computational load for subsequent layers
  2. It provides a form of translation invariance (the exact location of a feature becomes less important)
  3. It helps the network focus on the most important features
  4. It helps control overfitting by reducing the number of parameters

- Option A (introducing non-linearity) is incorrect. Non-linearity is introduced by activation functions like ReLU, not by pooling layers.

- Option C (normalizing activations) is incorrect. Normalization is typically done by batch normalization layers, not pooling layers.

- Option D (connecting all neurons) is incorrect. This describes fully connected (dense) layers, not pooling layers.
        """
    },
    {
        "id": 9,
        "category": "Reinforcement Learning",
        "question": "What is the difference between on-policy and off-policy learning in reinforcement learning?",
        "options": [
            "A. On-policy learning uses a value function, while off-policy learning uses a policy function",
            "B. On-policy learning evaluates and improves the same policy that's used to make decisions, while off-policy learning improves a different policy than the one used to make decisions",
            "C. On-policy learning is used for discrete action spaces, while off-policy learning is used for continuous action spaces",
            "D. On-policy learning requires more computational resources than off-policy learning"
        ],
        "correct_answer": "B",
        "explanation": """
The difference between on-policy and off-policy learning in reinforcement learning is that on-policy learning evaluates and improves the same policy that's used to make decisions, while off-policy learning improves a different policy than the one used to make decisions.

- In on-policy learning, the agent learns the value of the policy being used to make decisions (the behavior policy). Examples include SARSA and REINFORCE algorithms.

- In off-policy learning, the agent learns the value of a different policy (the target policy) than the one being used to make decisions (the behavior policy). This allows the agent to learn from data generated by old policies or even by humans. Examples include Q-learning and Deep Q-Networks (DQN).

- The key advantage of off-policy learning is that it can reuse data more efficiently, learning from experiences generated by any policy, not just the current one.

- Option A is incorrect because both on-policy and off-policy methods can use value functions and policy functions.

- Option C is incorrect because both approaches can be applied to both discrete and continuous action spaces.

- Option D is incorrect because the computational requirements depend on the specific algorithms, not on whether they are on-policy or off-policy.
        """
    },
    {
        "id": 10,
        "category": "Model Optimization",
        "question": "Which of the following is NOT a common regularization technique used to prevent overfitting in machine learning models?",
        "options": [
            "A. L1 regularization (Lasso)",
            "B. L2 regularization (Ridge)",
            "C. Dropout",
            "D. Batch normalization"
        ],
        "correct_answer": "D",
        "explanation": """
Batch normalization is NOT primarily a regularization technique, although it can have some regularizing effects.

- Batch normalization is a technique used to improve the training of neural networks by normalizing the activations of each layer. It helps address the internal covariate shift problem and can speed up training by allowing higher learning rates. While it can have some regularizing effects (reducing the need for dropout in some cases), its primary purpose is not regularization but rather improving and stabilizing the training process.

- L1 regularization (Lasso) (option A) is a regularization technique that adds the sum of the absolute values of the weights to the loss function. This encourages sparse models by pushing some weights exactly to zero, effectively performing feature selection.

- L2 regularization (Ridge) (option B) is a regularization technique that adds the sum of the squared weights to the loss function. This discourages large weights and tends to distribute the weight values more evenly.

- Dropout (option C) is a regularization technique specific to neural networks where randomly selected neurons are ignored during training. This prevents neurons from co-adapting too much and forces the network to learn more robust features.

Other common regularization techniques not mentioned in the options include:
1. Early stopping
2. Data augmentation
3. Weight decay
4. Elastic Net (combination of L1 and L2)
        """
    }
]
