import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set a better style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")
print("Dataset loaded successfully!\n")

# Explore the dataset
print("First 5 rows of data:")
print(df.head(), "\n")
print("Info about dataset:")
print(df.info(), "\n")
print("Summary statistics:")
print(df.describe(), "\n")

# Add Study Hours column
np.random.seed(42)
df['study_hours'] = np.random.randint(1, 10, size=len(df))
print("Dataset with Study Hours column:")
print(df.head(), "\n")

# Visualization 1: Study Hours vs Math Score (with regression line)
plt.figure(figsize=(8,6))
sns.scatterplot(x="study_hours", y="math score", data=df, hue="gender", palette="coolwarm", s=80)
sns.regplot(x="study_hours", y="math score", data=df, scatter=False, color='black', line_kws={'linewidth':2})
plt.title("📊 Study Hours vs Math Score")
plt.xlabel("Study Hours per Day")
plt.ylabel("Math Score")
plt.legend(title="Gender")
plt.show()

# Visualization 2: Correlation Heatmap
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="vlag", linewidths=0.5, cbar=True)
plt.title("🔥 Correlation Heatmap of Features")
plt.show()

# Visualization 3: Average Math Score by Test Preparation Course
avg_score_tp = df.groupby("test preparation course")["math score"].mean().reset_index()
sns.barplot(x="test preparation course", y="math score", data=avg_score_tp, palette="viridis")
plt.title("Average Math Score by Test Preparation Course")
plt.ylabel("Average Math Score")
plt.xlabel("Test Preparation Course")
plt.show()

# Visualization 4: Average Math Score by Gender
avg_score_gender = df.groupby("gender")["math score"].mean().reset_index()
sns.barplot(x="gender", y="math score", data=avg_score_gender, palette="pastel")
plt.title("Average Math Score by Gender")
plt.ylabel("Average Math Score")
plt.xlabel("Gender")
plt.show()

# Predicting Math Scores using Study Hours
X = df[['study_hours']]   # feature
y = df['math score']      # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Visualization 5: Predicted vs Actual Scores
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='purple', alpha=0.7, s=80)
plt.plot([0, 100], [0, 100], 'r--', linewidth=2)
plt.title("Predicted vs Actual Math Scores")
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.show()

# Model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Performance:\nMSE: {mse:.2f}\nR² Score: {r2:.2f}\n")

print("Analysis complete! 🎉")
