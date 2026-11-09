# 🚗 Car Industry Cluster Predictor

A Machine Learning project that applies **K-Means and DBSCAN clustering** and classification algorithms to analyze car industry data and predict cluster assignments for new records.

The project includes data integration using SQLite, clustering, classification, hyperparameter tuning, model evaluation, input validation, and an interactive Streamlit application.

## 📌 Project Overview

The main objective is to analyze car industry data and identify groups of vehicle and sales records using unsupervised machine learning.

After clustering, the generated K-Means cluster labels are used as target values to train classification models. The best-performing model is saved and integrated into a Streamlit application for predictions on new data.

## 🎯 Project Objectives

- Integrate multiple Excel datasets using SQL JOIN operations.
- Store and manage data using SQLite.
- Apply K-Means and DBSCAN clustering algorithms.
- Train three classification algorithms.
- Perform hyperparameter tuning.
- Compare model performance.
- Save the selected model using Joblib.
- Validate new user inputs.
- Predict cluster assignments through Streamlit.

## 📂 Dataset Information

The project uses three main datasets:

| Dataset | Records | Description |
|---|---:|---|
| Companies | 20 | Company information |
| Models | 400 | Vehicle model information |
| Sales | 2,500 | Car sales records |

The three datasets are joined to create a combined dataset containing **2,500 records and 15 columns**.

### Main Features

- ModelYear
- Year
- UnitsSold
- IllustrativePriceUSD_Min
- IllustrativePriceUSD_Max

## 🧠 Machine Learning Algorithms

### 1. Clustering Algorithms

- K-Means Clustering
- DBSCAN Clustering

K-Means cluster labels are used as the target variable for the classification stage.

### 2. Classification Algorithms

- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

### 3. Hyperparameter Tuning

GridSearchCV is used to identify suitable model parameters. Macro F1-score is used as the cross-validation scoring metric.

## 📊 Model Performance

| Classification Algorithm | Test Accuracy | CV Macro F1 |
|---|---:|---:|
| Decision Tree | 99.8% | 0.991994 |
| Random Forest | 99.8% | 0.983933 |
| K-Nearest Neighbors | 99.8% | 0.994299 |

**Selected Model: K-Nearest Neighbors (KNN)**

KNN was selected because it achieved the highest cross-validation Macro F1-score while maintaining 99.8% test accuracy.

> Note: The classification target consists of K-Means-generated cluster labels. Therefore, these evaluation results measure how well the classifier reproduces the clustering assignments, not independently verified real-world vehicle categories.

## 💾 Model Saving and Deployment

The selected KNN pipeline is saved using Joblib.

Model file:

`car_industry_knn_pipeline.joblib`

The saved bundle contains:
- Trained KNN pipeline
- Required input features
- Target information
- Selected algorithm
- Best hyperparameters
- Cluster labels
- Observed training feature ranges

## 🌐 Streamlit Application

The project includes an interactive Streamlit application.

### Application Features

- Enter vehicle and sales information.
- Validate numerical inputs.
- Check values against observed training ranges.
- Predict the K-Means cluster.
- Display the predicted cluster.
- Display model probability estimates.

## 🗂️ Project Structure

```text
Car-Industry-Cluster-Predictor/
│
├── app.py
├── requirements.txt
├── car_industry_knn_pipeline.joblib
└── README.md
```

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/malikshehryarali40/Car-Industry-Cluster-Predictor.git
```

### Step 2: Open the Project Folder

```bash
cd Car-Industry-Cluster-Predictor
```

### Step 3: Create a Virtual Environment

```bash
python3 -m venv venv
```

### Step 4: Activate the Environment

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available locally at:

`http://localhost:8501`

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SQLite
- Joblib
- Streamlit
- Google Colab
- Git and GitHub

## ⚠️ Important Limitations

- The model predicts K-Means-generated cluster labels rather than independently defined business categories.
- The input validation ranges are based on the minimum and maximum values observed in the training dataset.
- Model predictions depend on the quality and coverage of the original dataset.
- The reported classification accuracy should not be interpreted as proof of real-world business prediction accuracy.

## 👨‍💻 Author

**Shehryar Ali**

Machine Learning Project

GitHub: [@malikshehryarali40](https://github.com/malikshehryarali40)

---

⭐ If you find this project useful, consider giving the repository a star.
