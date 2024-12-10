# Burnout-Prediction
Description:   This repository contains a project analyzing the relationship between daily lifestyle habits (study, sleep, physical activity, etc.) and student academic performance and stress levels. Using machine learning and SQL, it predicts burnout risks and provides actionable insights for balanced living for students.
Predicting Academic Performance and Stress Management in Students

This project explores how daily lifestyle habits impact academic performance (GPA) and stress levels in students. By leveraging machine learning models and clustering techniques, we provide actionable insights for improved well-being and academic success.

Dataset

The dataset used for this project is publicly available on Kaggle:
Student Lifestyle Dataset

The dataset includes features such as:

GPA (academic performance)
Stress levels (encoded as Low: 0, Moderate: 0.5, High: 1)
Daily habits: study hours, sleep hours, physical activities, social interactions, and extracurricular activities.
Setup Instructions

1. Creating the Virtual Environment
To ensure all dependencies are isolated, create a virtual environment:

python -m venv venv

2. Activating the Virtual Environment
Activate the virtual environment with the following command:

On Mac/Linux:
source ./venv/bin/activate

On Windows:
.\venv\Scripts\activate

3. Installing Required Libraries
Install the necessary libraries from the requirements.txt file:

pip install -r requirements.txt

Running the Project

1. Exploratory Data Analysis and Visualizations
Execute the eda_and_visualizations.ipynb notebook to:

Generate boxplots and violin plots.
Analyze correlations among lifestyle habits and stress levels.
2. Training and Evaluation
Run the predictive_modeling.ipynb notebook to train the regression and clustering models:

Regression: Predict GPA and stress levels.
Clustering: Identify lifestyle-based student groups using PCA and KMeans.
Key Features

Regression Analysis: Gradient Boosting Regressor for predicting stress levels and GPA.
Clustering: PCA + KMeans to categorize students based on lifestyle patterns.
Visualization: Interactive plots to reveal relationships among habits, GPA, and stress.
Insights and Recommendations

Study Habits: Students with balanced study and non-academic activities show lower stress levels.
Sleep Patterns: Reduced sleep correlates with higher stress but has minimal direct impact on GPA.
Lifestyle Clusters:
Cluster 0: Balanced habits, low to moderate stress.
Cluster 1: High academic focus, elevated stress.
Cluster 2: Socially active, low stress, but lower academic focus.
