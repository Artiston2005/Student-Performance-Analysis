# Student Performance Analysis 📊

## 1. Project Overview
This project performs an in-depth analysis of student academic performance based on various demographic and preparatory factors. The primary goal is to identify key indicators that influence student scores in math, reading, and writing. A predictive model is also built using Linear Regression to forecast math scores based on randomly generated study hours.

---

## 2. Dataset
The dataset used is `StudentsPerformance.csv`, sourced from Kaggle. It contains 1,000 records of student data with features such as parental education level, lunch type, and test preparation.

* **Source:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/spscientist/students-performance-in-exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
* **License:** CC0: Public Domain

---

## 3. Tools & Libraries
* **Python:** The core language for the analysis.
* **Pandas:** For data manipulation and cleaning.
* **Matplotlib & Seaborn:** For data visualization.
* **Scikit-learn:** For building the Linear Regression model.

---

## 4. How to Run This Project
1.  Clone this repository to your local machine.
2.  Ensure you have Python and the required libraries installed by running:
    ```bash
    pip install -r requirements.txt
    ```
3.  Execute the Python script:
    ```bash
    python Student_performance_analysis.py
    ```
The script will print dataset information to the console and generate several visualizations.

---

## 5. Key Findings
* **Test Preparation is Key:** Students who completed the test preparation course consistently scored higher across all subjects. This was the most significant indicator of improved performance.
* **Parental Education Matters:** There is a positive correlation between the parental level of education and student scores, with students of parents holding higher degrees performing better on average.
* **Predictive Model:** The linear regression model demonstrated a positive correlation between study hours and math scores, confirming that more study time can lead to better performance.

---

## 6. Contact
Created by Ashwin Yadav - [ashwinyadav2408@gmail.com](mailto:ashwinyadav2408@gmail.com) - [LinkedIn](https://www.linkedin.com/in/ashwin-yadav-1704a1248/)
