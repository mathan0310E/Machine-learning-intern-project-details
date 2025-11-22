# Iris Machine Learning Project – Level 1 (Codveda Internship)

This repository contains my Level 1 Machine Learning internship project for **Codveda**, built using the **Iris dataset**.  
The project focuses on three core tasks:

1. Data Preprocessing  
2. Simple Linear Regression  
3. K-Nearest Neighbors (KNN) Classification  

All experiments are implemented using **Python** and **Scikit-Learn**.

---

## 1. Dataset Description

The **Iris dataset** consists of:

- **150 samples**
- **3 classes**: Iris-setosa, Iris-versicolor, Iris-virginica  
- **4 numerical features**:
  - Sepal Length  
  - Sepal Width  
  - Petal Length  
  - Petal Width  

This dataset is widely used as an introduction to supervised learning and classification.

---

## 2. Tasks Performed

### 2.1 Data Preprocessing

Steps followed:

- Checked for missing values  
- Handled numerical data (mean imputation if required)  
- Encoded the target label (`species`) using **LabelEncoder**  
- Split the dataset into:
  - **80% Training Data**
  - **20% Testing Data**
- Standardized the numerical features using **StandardScaler**  
  (important for distance-based algorithms like KNN)

---

### 2.2 Simple Linear Regression

**Objective:**  
Predict the continuous feature **`petal_length`** using other numerical features (e.g. `sepal_length`, `sepal_width`, `petal_width`).

**Model Used:**  
- `LinearRegression` from `sklearn.linear_model`

**Evaluation Metrics:**

- **Mean Squared Error (MSE)**
- **R-squared (R²)**

A scatter plot of **True vs Predicted Petal Length** was also generated to visualize how well the model fits.

---

### 2.3 K-Nearest Neighbors (KNN) Classification

**Objective:**  
Classify flowers into one of the three species based on the four numeric measurements.

**Model Used:**  
- `KNeighborsClassifier` from `sklearn.neighbors`

**Process:**

- Standardized features (from preprocessing step)
- Trained KNN with different K values: **1, 3, 5, 7, 9**
- Selected the **best K** based on highest accuracy on test data

**Evaluation Metrics:**

- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score for each class)

A line plot of **K vs Accuracy** was used to study how the number of neighbors affects model performance.

---


Run the Notebook
Open main_code.ipynb in:
Jupyter Notebook
or
Google Colab (upload the notebook + dataset)

Run all cells in order to reproduce:
Data preprocessing
Linear Regression training & evaluation
KNN Classification training & evaluation


Tools & Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-Learn


Learning Outcomes
From this project, I learned:
How to preprocess data for machine learning
How to apply regression for continuous prediction
How to implement and tune KNN classification
How to evaluate models using metrics and visualizations
This project forms the foundation for more advanced ML tasks in future internship levels.

Acknowledgement
This project was completed as part of the Machine Learning Internship at Codveda.
I am thankful to the team for providing guidance and structured tasks to learn the basics of Machine Learning.

---

## ✅ `requirements.txt` (create this file in your repo)

```txt
pandas
numpy
matplotlib
scikit-learn

