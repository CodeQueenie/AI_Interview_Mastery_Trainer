# AI Interview Mastery Trainer

An interactive training tool designed to help users master essential skills for AI Engineer interviews.

## Key Features

- **Coding Practice**: Algorithm and data structure challenges with hints, solutions, and complexity analysis
- **Theory Questions**: Multiple-choice questions on AI/ML fundamentals with detailed explanations
- **Algorithm Design**: Design algorithms for real-world AI scenarios with guided approaches
- **Study Guides**: Comprehensive learning materials for each topic with practical interview strategies
- **Progress Tracking**: Dashboard to monitor your performance and identify areas for improvement
- **Personalized Recommendations**: Get tailored study suggestions based on your performance

## Recent Updates

- Added comprehensive study guides for all question types
- Implemented pre-question checklists to guide preparation
- Enhanced study progress tracker with visual indicators
- Improved navigation between study materials and practice questions

## Getting Started

1. Clone this repository:
```
git clone https://github.com/CodeQueenie/AI_Interview_Mastery_Trainer.git
cd AI_Interview_Mastery_Trainer
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run src/app.py
```

## Project Structure

```
AI_Interview_Mastery_Trainer/
├── data/
│   ├── coding_questions.py        # Coding problems with solutions
│   ├── theory_questions.py        # AI/ML theoretical questions
│   ├── algorithm_design_questions.py  # Algorithm design problems
│   └── study_guides.py            # Study materials for all topics
├── src/
│   ├── app.py                     # Main Streamlit application
│   ├── dashboard.py               # Progress tracking dashboard
│   ├── utils.py                   # Utility functions
│   └── visualizations.py          # Data visualization utilities
├── images/                        # Screenshots and visual assets
├── logs/                          # Application logs
├── session_data/                  # User session storage
├── run.py                         # Application launcher
└── requirements.txt               # Project dependencies
```

## License

This project is licensed under the MIT License with attribution requirements.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
