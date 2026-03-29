## Customer Churn Prediction System

## 📌 Project Overview

This project is an end-to-end Machine Learning application that predicts whether a customer is likely to churn (leave a service) based on their demographic and service usage data.

The system not only provides churn predictions but also gives actionable business insights to help reduce customer attrition.

---

## 🎯 Objective

To build a predictive model that identifies customers at high risk of churn, enabling businesses to take proactive retention measures.

---

## 🧠 Features

* 🔍 Predicts customer churn (Yes/No)
* 📊 Displays churn probability score
* ⚠️ Identifies high-risk customers
* 💡 Provides business recommendations
* 🖥️ Interactive user interface using Streamlit

---

## 🏗️ Tech Stack

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Model:** Logistic Regression, Random Forest
* **Frontend/UI:** Streamlit
* **Deployment:** Local (can be extended to cloud)

---

## 📂 Project Structure

```
customer-churn-project/
│
├── data/
│   └── churn.csv
│
├── preprocessing.py      # Data cleaning & encoding
├── model.py              # Model training & evaluation
├── train.py              # Training pipeline
├── app.py                # Streamlit UI
├── model.pkl             # Saved trained model
└── README.md
```

---

## 🔄 Workflow

1. Data Collection (Kaggle dataset)
2. Data Preprocessing (cleaning, encoding)
3. Exploratory Data Analysis (EDA)
4. Model Training & Evaluation
5. Model Saving (pickle)
6. Deployment using Streamlit UI

---

## ▶️ How to Run the Project

### Step 1: Clone the repository

```
git clone <your-repo-link>
cd customer-churn-project
```

### Step 2: Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```
pip install -r requirements.txt
```

### Step 4: Train the model

```
python train.py
```

### Step 5: Run the application

```
streamlit run app.py
```

---

## 📊 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

Special focus is given to **Recall**, as identifying potential churn customers is critical.

---

## 💡 Business Impact

This system helps businesses:

* Identify customers likely to churn
* Take preventive actions (offers, discounts, engagement)
* Improve customer retention
* Increase revenue

---

## 🚀 Future Enhancements

* Add XGBoost for better performance
* Deploy on cloud (AWS / Render / Streamlit Cloud)
* Add real-time API integration
* Improve UI/UX design

---

## 🙌 Acknowledgements

* Dataset sourced from Kaggle
* Built as part of a Machine Learning project


