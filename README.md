# Student Performance ML Project

A **Streamlit** web app that predicts student performance and demonstrates both supervised and unsupervised machine learning techniques using exam scores data.  

This project allows you to:  
- Train and compare **Logistic Regression** and **Decision Tree** models.  
- Perform **KMeans** and **Hierarchical Clustering** on student scores.  
- Input custom student data to predict performance category.  

---

## Project Structure


.
├── app.py # Main Streamlit app
├── models.py # Supervised and unsupervised ML models
├── preprocessing.py # Data loading and preprocessing
├── data/
│ └── exams.csv # Student exam dataset
├── requirements.txt # Required Python libraries
└── README.md # Project documentation


---

## Features

### Supervised Learning
- Train **Logistic Regression** and **Decision Tree** classifiers.  
- View metrics like **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **Confusion Matrix**.  
- Compare model performance using bar plots.  

### Unsupervised Learning
- Perform **KMeans** and **Hierarchical Clustering**.  
- Automatically select the best number of clusters using **Silhouette Score**.  
- Visualize clusters using scatter plots.  

### Prediction
- Input student details and exam scores.  
- Get predicted **Performance Category** from both models.  

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>

Create and activate virtual environment

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

Install dependencies

pip install -r requirements.txt
Usage

Run the Streamlit app:

streamlit run app.py

Open the URL provided (usually http://localhost:8501) and use the sidebar to navigate between:

Supervised Learning

Unsupervised Learning

Prediction

Dataset

The dataset exams.csv should have:

Column	Type	Description
gender	categorical	Student gender
race/ethnicity	categorical	Student race/ethnicity group
parental level of education	categorical	Parent's education level
lunch	categorical	Type of lunch (standard/free)
test preparation course	categorical	Completed preparation course
math score	numerical	Score in Math
reading score	numerical	Score in Reading
writing score	numerical	Score in Writing
Dependencies

Python 3.8+

pandas

numpy

scikit-learn

matplotlib

seaborn
