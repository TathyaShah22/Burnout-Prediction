# %%
# Install required libraries
import subprocess
import sys


libraries = [
    "pandas", "numpy", "matplotlib", "seaborn", "scipy", 
    "statsmodels", "scikit-learn", "imbalanced-learn", 
    "sqlite3", "sqlalchemy", "os", "pathlib"
]

def install_library(lib):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
    except Exception as e:
        print(f"Failed to install {lib}: {e}")
pip_libraries = [
    "pandas", "numpy", "matplotlib", "seaborn", 
    "scipy", "statsmodels", "scikit-learn", 
    "imbalanced-learn", "sqlalchemy"
]

for lib in pip_libraries:
    install_library(lib)

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import sqlite3
from sqlalchemy import create_engine
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor


# %%
stu_life = pd.read_csv('student_lifestyle_dataset.csv')

stu_life.rename(columns={'Study_Hours_Per_Day': 'Study/Day', 
                        'Sleep_Hours_Per_Day': 'Sleep/Day',
                        'Social_Hours_Per_Day': 'Social/Day',
                        'Physical_Activity_Hours_Per_Day': 'Physical/Day',
                        'Extracurricular_Hours_Per_Day': 'Extracurricular/Day'}, 
                        inplace=True)
stress_mapping = {'Low': 0, 'Moderate': 0.5, 'High': 1}
stu_life['Stress_Level_Encoded'] = stu_life['Stress_Level'].map(stress_mapping)

#STRESS LEVEL ANALYSIS

numeric_features = ['Study/Day', 'Sleep/Day', 'Social/Day', 'Physical/Day', 'Extracurricular/Day']
scaler = StandardScaler()
normalized_data = stu_life.drop(['Student_ID','Stress_Level', 'GPA'],axis = 1)
normalized_data[numeric_features] = scaler.fit_transform(stu_life[numeric_features])
normalized_data


# %%
# Boxplots for Stress Levels
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=normalized_data, x="Stress_Level_Encoded", y=feature, palette="Set2")
    plt.title(f"{feature} by Stress Level")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# Violin Plot for GPA by Stress Level
plt.figure(figsize=(8, 6))
sns.violinplot(data=stu_life, x="Stress_Level", y="GPA", palette="muted")
plt.title("GPA Trends by Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("GPA")
plt.show()

# %%
# Correlation Heatmap
plt.figure(figsize=(10, 6))
correlation_matrix = normalized_data[numeric_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Normalized Features")
plt.show()

# %%
#UNDERSTANDING COORELATION BETWEEN THE PREDICTORS/VARIABLES/FEATURES
correlation_matrix = stu_life[numeric_features].corr()
print(correlation_matrix)


# %%
#NEW METRICS FOR PREDICTION MODELS AS PER THE COORELATION BETWEEN VARIABLES
#Study/Sleep
normalized_data['Study/Sleep'] = normalized_data['Study/Day'] / normalized_data['Sleep/Day']
# NonStudy 
normalized_data['NonStudy'] = (
    normalized_data.sum(axis=1) - normalized_data['Study/Day'] - normalized_data['Sleep/Day']
)
# Ratio of NonStudy activities to Study time
normalized_data['NonStudy/Study'] = normalized_data['NonStudy'] / normalized_data['Study/Day']
# Ratio of Sleep to NonStudy activities
normalized_data['Sleep/NonStudy'] = normalized_data['Sleep/Day'] / normalized_data['NonStudy']
#Ratio of Social to NonStudy activities
normalized_data['Social/NonStudy'] = normalized_data['Social/Day']/normalized_data['NonStudy']
normalized_data


# %%
#REGRESSION AND PREDICTION
models = {
    "Model 1: Study/Sleep Only": ['Study/Sleep', 'Extracurricular/Day', 'Social/Day', 'Physical/Day'],
    "Model 2: NonStudy/Study Only": ['NonStudy/Study', 'Sleep/Day', 'Extracurricular/Day', 'Social/Day', 'Physical/Day'],
    "Model 3: Sleep/NonStudy Only": ['Sleep/NonStudy', 'Physical/Day', 'Extracurricular/Day', 'Social/Day'],
    "Model 4: All Metrics": ['Study/Sleep', 'NonStudy/Study', 'Sleep/NonStudy', 'Social/Day', 'Extracurricular/Day', 'Physical/Day']
}

results = {}
for model_name, features in models.items():
    X = normalized_data[features]
    y = normalized_data['Stress_Level_Encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.15)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    results[model_name] = {"R^2 Score": r2, "Mean Squared Error": mse}
    
results_df = pd.DataFrame(results).T
print(results_df)


# %%
#CROSS VALIDATION
cross_val_results = {}

for model_name, features in models.items():
    X = normalized_data[features]
    y = normalized_data['Stress_Level_Encoded']

    model = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.15)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cross_val_results[model_name] = {
        "Mean R^2 Score": np.mean(cv_scores),
        "Standard Deviation": np.std(cv_scores)
    }

cross_val_results_df = pd.DataFrame(cross_val_results).T

print("Cross-Validation Results for Models:")
print(cross_val_results_df)

# %%
# Feature importance for GradientBoostingRegressor
# Select the predictors and model for Model 2
X_model_2 = normalized_data[models["Model 2: NonStudy/Study Only"]]
y_model_2 = normalized_data['Stress_Level_Encoded']

model_2 = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.15)
model_2.fit(X_model_2, y_model_2)

y_pred_model_2 = model_2.predict(X_model_2)

importance = model_2.feature_importances_
for feature, imp in zip(X_model_2.columns, importance):
    print(f"{feature} importance: {imp}")

# %%
#SYNTHETIC DATASET
# Generate synthetic data
np.random.seed(42)
n_samples = 500

synthetic_data = pd.DataFrame({
    'NonStudy/Study': np.random.uniform(0, 10, n_samples),
    'Sleep/Day': np.random.uniform(4, 10, n_samples),
    'Extracurricular/Day': np.random.uniform(0, 5, n_samples),
    'Social/Day': np.random.uniform(0, 8, n_samples),
    'Physical/Day': np.random.uniform(0, 3, n_samples),
})

# Simulate stress level (target variable)
synthetic_data['Stress_Level_Encoded'] = (
    -0.2 * synthetic_data['NonStudy/Study'] +
    0.3 * synthetic_data['Sleep/Day'] -
    0.1 * synthetic_data['Extracurricular/Day'] +
    0.4 * synthetic_data['Social/Day'] -
    0.3 * synthetic_data['Physical/Day'] +
    np.random.normal(0, 1, n_samples)  # Add some noise
)

normalized_data = (synthetic_data - synthetic_data.min()) / (synthetic_data.max() - synthetic_data.min())

features = ['NonStudy/Study', 'Sleep/Day', 'Extracurricular/Day', 'Social/Day', 'Physical/Day']
X = normalized_data[features]
y = normalized_data['Stress_Level_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=3, learning_rate=0.05)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

results = {"Model 2 on Synthetic Data": {"R^2 Score": r2, "Mean Squared Error": mse}}
results_df = pd.DataFrame(results).T
print(results_df)
synthetic_data


# %%
#SYNTHETIC PLOT
# Predictions for synthetic data (Model 2)
y_pred_synthetic = model.predict(X)

# Scatter Plot: Actual vs. Predicted Stress Levels for Synthetic Data
plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred_synthetic, alpha=0.6, color='blue')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label="Perfect Prediction")
plt.title("Actual vs. Predicted Stress Levels (Synthetic Data - Model 2)")
plt.xlabel("Actual Stress Levels")
plt.ylabel("Predicted Stress Levels")
plt.legend()
plt.show()

# %%
#NEED OBSERVATIONS FOR PLOT
#KMEANS CLUSTERING

# Map stress levels to numerical values
stress_mapping = {'Low': 0, 'Moderate': 0.5, 'High': 1}
stu_life['Stress_Level_Encoded'] = stu_life['Stress_Level'].map(stress_mapping)

# Normalize the data
numeric_features = ['Study/Day', 'Sleep/Day', 'Social/Day', 'Physical/Day', 'Extracurricular/Day']
scaler = StandardScaler()
normalized_data = stu_life.drop(['Student_ID', 'Stress_Level', 'GPA'], axis=1)
normalized_data[numeric_features] = scaler.fit_transform(stu_life[numeric_features])

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
pca_data = pca.fit_transform(normalized_data[numeric_features])

# Apply KMeans to the PCA-reduced data
kmeans = KMeans(n_clusters=3, random_state=42)
normalized_data['PCA_KMeans_Cluster'] = kmeans.fit_predict(pca_data)

# Visualize the clusters
plt.figure(figsize=(10, 6))
for cluster in range(3):
    cluster_data = normalized_data[normalized_data['PCA_KMeans_Cluster'] == cluster]
    plt.scatter(pca_data[normalized_data['PCA_KMeans_Cluster'] == cluster, 0],
                pca_data[normalized_data['PCA_KMeans_Cluster'] == cluster, 1],
                label=f'Cluster {cluster}', s=50)

# Add labels and title
plt.title('PCA + KMeans Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Display cluster counts
cluster_counts = normalized_data['PCA_KMeans_Cluster'].value_counts()
print("Cluster Counts:\n", cluster_counts)

# %%
#SQL QUERIES

data = pd.read_csv("student_lifestyle_dataset.csv")

conn = sqlite3.connect("student_lifestyle.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS StudentActivities")

# Create the StudentActivities table
cursor.execute('''
CREATE TABLE StudentActivities (
    Student_ID INTEGER PRIMARY KEY,
    Study_Hours_Per_Day REAL,
    Extracurricular_Hours_Per_Day REAL,
    Sleep_Hours_Per_Day REAL,
    Social_Hours_Per_Day REAL,
    Physical_Activity_Hours_Per_Day REAL,
    GPA REAL,
    Stress_Level REAL
);
''')

data.to_sql("StudentActivities", conn, if_exists="append", index=False)
conn.commit()


queries = {
    "Average GPA and Key Metrics": '''
        SELECT 
            AVG(GPA) AS avg_gpa,
            AVG(Study_Hours_Per_Day) AS avg_study,
            AVG(Sleep_Hours_Per_Day) AS avg_sleep,
            AVG(Extracurricular_Hours_Per_Day) AS avg_extracurricular,
            AVG(Social_Hours_Per_Day) AS avg_social,
            AVG(Physical_Activity_Hours_Per_Day) AS avg_physical
        FROM StudentActivities;
    ''',

    "GPA by Study Categories": '''
        SELECT 
            CASE 
                WHEN Study_Hours_Per_Day < 2 THEN 'Low Study'
                WHEN Study_Hours_Per_Day BETWEEN 2 AND 5 THEN 'Moderate Study'
                ELSE 'High Study'
            END AS study_category,
            AVG(GPA) AS avg_gpa
        FROM StudentActivities
        GROUP BY study_category;
    ''',

    "Average GPA by Sleep Levels": '''
        SELECT 
            CASE 
                WHEN Sleep_Hours_Per_Day < 6 THEN 'Low Sleep'
                WHEN Sleep_Hours_Per_Day BETWEEN 6 AND 8 THEN 'Moderate Sleep'
                ELSE 'High Sleep'
            END AS sleep_category,
            AVG(GPA) AS avg_gpa
        FROM StudentActivities
        GROUP BY sleep_category;
    ''',

    "Combined Effect of Sleep and Study": '''
        SELECT 
            CASE 
                WHEN Sleep_Hours_Per_Day < 6 THEN 'Low Sleep'
                ELSE 'High Sleep'
            END AS sleep_category,
            CASE 
                WHEN Study_Hours_Per_Day < 2 THEN 'Low Study'
                ELSE 'High Study'
            END AS study_category,
            AVG(GPA) AS avg_gpa
        FROM StudentActivities
        GROUP BY sleep_category, study_category;
    ''',

    "GPA by Non-Academic Activities": '''
        SELECT 
            CASE 
                WHEN Extracurricular_Hours_Per_Day > 2 THEN 'High Extracurricular'
                ELSE 'Low Extracurricular'
            END AS extracurricular_category,
            AVG(GPA) AS avg_gpa
        FROM StudentActivities
        GROUP BY extracurricular_category;
    ''',

    "Top Students by GPA": '''
        SELECT * 
        FROM StudentActivities
        ORDER BY GPA DESC
        LIMIT 10;
    ''',

    "Students with the Most Balanced Schedule": '''
        SELECT * 
        FROM StudentActivities
        WHERE ABS(Study_Hours_Per_Day - Sleep_Hours_Per_Day) < 1
        ORDER BY GPA DESC
        LIMIT 10;
    ''',

    "Study-to-Sleep Ratio": '''
        SELECT 
            Study_Hours_Per_Day / Sleep_Hours_Per_Day AS study_to_sleep_ratio,
            GPA
        FROM StudentActivities;
    ''',

    "Effect of Non-Study Activities on GPA": '''
        SELECT 
            Extracurricular_Hours_Per_Day + Social_Hours_Per_Day + Physical_Activity_Hours_Per_Day AS non_study_total,
            AVG(GPA) AS avg_gpa
        FROM StudentActivities
        GROUP BY non_study_total
        ORDER BY avg_gpa DESC;
    ''',

    "GPA by Stress Levels": '''
        SELECT 
            CASE 
                WHEN Stress_Level < 0.33 THEN 'Low Stress'
                WHEN Stress_Level BETWEEN 0.33 AND 0.66 THEN 'Moderate Stress'
                ELSE 'High Stress'
            END AS stress_category,
            COUNT(*) AS student_count,
            AVG(GPA) AS avg_gpa
        FROM StudentActivities
        GROUP BY stress_category;
    '''
}



for title, query in queries.items():
    print(f"{title}:\n")
    result = pd.read_sql_query(query, conn)
    print(result)
    print("\n" + "-"*40 + "\n")
conn.close()


# %%



