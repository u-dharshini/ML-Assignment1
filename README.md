# Student Performance ML Project

A **Streamlit** web app that predicts student performance and demonstrates both supervised and unsupervised machine learning techniques using exam scores data.  

This project allows you to:  
- Train and compare **Logistic Regression** and **Decision Tree** models.  
- Perform **KMeans** and **Hierarchical Clustering** on student scores.  
- Input custom student data to predict performance category.  

---

## Project Structure


Student-Performance-ML-Project/
├── app.py # Main Streamlit app
├── models.py # Supervised and unsupervised ML models
├── preprocessing.py # Data loading and preprocessing
├── data/
│ └── exams.csv # Student exam dataset
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## Features

### Supervised Learning
- Train **Logistic Regression** and **Decision Tree** classifiers.  
- Evaluate models with **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **Confusion Matrix**.  
- Compare model performance using bar plots.  

### Unsupervised Learning
- Perform **KMeans** and **Hierarchical Clustering**.  
- Automatically select the best number of clusters using **Silhouette Score**.  
- Visualize clusters using scatter plots.  

### Prediction
- Input student details and exam scores.  
- Get predicted **Performance Category** from both models.  

---

## Dataset

The dataset `exams.csv` should have the following columns:

| Column                       | Type       | Description                        |
|-------------------------------|-----------|------------------------------------|
| gender                        | categorical | Student gender                     |
| race/ethnicity                | categorical | Student race/ethnicity group       |
| parental level of education   | categorical | Parent's education level           |
| lunch                         | categorical | Type of lunch (standard/free)      |
| test preparation course       | categorical | Completed preparation course       |
| math score                    | numerical  | Score in Math                      |
| reading score                 | numerical  | Score in Reading                   |
| writing score                 | numerical  | Score in Writing                   |

**Note:** `Performance` is automatically created based on average score:
- `High`: 80+  
- `Medium`: 50–79  
- `Low`: <50  

---

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit

---
