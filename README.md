
# 🫀 Heart Disease Prediction - ML Pipeline

This project presents a full machine learning pipeline for analyzing, predicting, and visualizing heart disease risk using the **Heart Disease UCI Dataset**. The solution includes preprocessing, dimensionality reduction, feature selection, supervised and unsupervised learning, model optimization, and a web UI with deployment.

---

## 📌 Dataset
**Heart Disease UCI Dataset From Kaggle**  
🔗 [UCI Repository Link](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

---

## 🎯 Objectives
- Clean and preprocess raw data
- Apply dimensionality reduction (PCA)
- Select key features using statistical and ML techniques
- Train classification models:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - SVM
  - KNN
  - XGBoost
  - Gradient Boosting
- Apply clustering algorithms:
  - K-Means
  - Hierarchical Clustering
- Optimize model performance
- Build and deploy a real-time prediction app using Streamlit and Ngrok

---

## 🛠️ Tools & Libraries
- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `xgboost` (optional)
  - `Streamlit`, `Ngrok`
- **Techniques**:
  - PCA
  - RFE, Chi-Square
  - GridSearchCV, RandomizedSearchCV
  - joblib / pickle

---

## 🚀 Pipeline Overview

### 1. Data Preprocessing
- Handle missing values
- One-hot encode categorical variables
- Scale numerical features
- Exploratory Data Analysis (EDA)

### 2. Dimensionality Reduction - PCA
- Apply PCA
- Analyze explained variance
- Visualize results

### 3. Feature Selection
- Feature Importance (Random Forest, XGBoost)
- RFE, Chi-Square Test
- Select top features

### 4. Supervised Learning
Train & evaluate:
- Logistic Regression
- Decision Tree
- Random Forest
- SVM  
**Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC

### 5. Unsupervised Learning
- K-Means (with elbow method)
- Hierarchical Clustering (dendrograms)
- Compare clusters with labels

### 6. Hyperparameter Tuning
- Optimize models using GridSearchCV & RandomizedSearchCV

### 7. Model Export
- Save best models (`.pkl`)
- Ensure pipeline reproducibility

### 8. Streamlit Web UI [Bonus]
- Input health data
- Predict in real time
- Visualize trends

### 9. Ngrok Deployment [Bonus]
- Serve Streamlit locally
- Create public access via Ngrok

### 10. GitHub Upload
- Include all source files, scripts, and models
- Add a `requirements.txt`
- Provide documentation for setup and deployment

---

## 📁 File Structure

```
Heart_Disease_Project/
│
├── data/
│   └── heart_disease.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│
├── models/
│   └── final_model.pkl
│
├── ui/
│   └── app.py  # Streamlit UI
│
├── deployment/
│   └── ngrok_setup.txt
│
├── results/
│   └── evaluation_metrics.txt
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ✅ Final Deliverables
- 📊 Cleaned & transformed dataset
- 📉 PCA and feature selection outputs
- 🔍 Trained and optimized models
- 📈 Evaluation metrics and visualizations
- 💾 `.pkl` model files
- 🖥️ Streamlit app for live prediction [Bonus]
- 🌐 Ngrok deployment link [Bonus]
- 📂 GitHub repo with all code & instructions
---
## 👩🏽‍💻 Author
# Tasneem Mohamed Ahmed Mohamed Imam Badr
# 🔗 [LinkedIn Link](https://www.linkedin.com/in/tasneem-mohamed-714b5a2b1/)
