# Deposit-Predictor-
The goal of this project is to develop a classification model that predicts whether a bank client will subscribe to a term deposit based on demographic, historical, and call-related information. This project simulates a real-world retail banking use case and applies data science techniques to optimize marketing strategies.

---

### 🏢 Business Context

Retail banks invest heavily in telemarketing campaigns to sell term deposits—a critical source of their revenue. However, these campaigns are expensive. Therefore, **targeting the right customers** who are most likely to subscribe is crucial for cost-effectiveness.

---

### 📁 Dataset Description

The dataset is split into two files:

* **`train.csv`** – contains labeled data for training the model (with `subscribed` as the target variable).
* **`test.csv`** – contains data to generate predictions using the trained model.

---

### 📌 Data Dictionary

| Variable      | Description                                          |
| ------------- | ---------------------------------------------------- |
| `ID`          | Unique client ID                                     |
| `age`         | Age of the client                                    |
| `job`         | Type of job                                          |
| `marital`     | Marital status                                       |
| `education`   | Education level                                      |
| `default`     | Credit in default (yes/no)                           |
| `housing`     | Has housing loan (yes/no)                            |
| `loan`        | Has personal loan (yes/no)                           |
| `contact`     | Contact communication type (e.g., cellular)          |
| `month`       | Contact month                                        |
| `day_of_week` | Day of the week of contact                           |
| `duration`    | Duration of last contact (in seconds)                |
| `campaign`    | Number of contacts in this campaign                  |
| `pdays`       | Days since last contact (-1 if not contacted before) |
| `previous`    | Number of previous contacts                          |
| `poutcome`    | Outcome of previous marketing campaign               |
| `subscribed`  | Target: Did the client subscribe? (yes/no)           |

---

### ⚙️ Technologies Used

* **Python 3.x**
* **Pandas, NumPy** – Data Manipulation
* **Matplotlib, Seaborn** – Data Visualization
* **Scikit-learn** – Model Building
* **Logistic Regression, Decision Tree, Random Forest, XGBoost**
* **GridSearchCV** – Hyperparameter Tuning
* **Pickle** – Model Serialization
* **Jupyter Notebook** – Development Environment

---

### 🔍 Workflow

1. **Data Exploration & Cleaning**

   * Handle missing values
   * Encode categorical features
   * Feature engineering (e.g., converting duration to minutes)

2. **Exploratory Data Analysis (EDA)**

   * Visual insights into key features
   * Identify patterns related to subscription

3. **Model Building**

   * Baseline Logistic Regression
   * Ensemble models like Random Forest & XGBoost
   * Evaluation with Accuracy, Precision, Recall, F1-Score

4. **Prediction**

   * Predict outcomes on unseen `test.csv` data
   * Save results to a CSV file

---

### 📈 Results

The best model achieved:

* **Accuracy**: *90*
* **F1-Score**: *0.90120*


---

### ✅ Conclusion

This project showcases how **machine learning can reduce costs and improve ROI** for banks by focusing efforts on customers most likely to subscribe to term deposits.

---

### 🤝 Acknowledgment

This project is part of the **Final Assignment** in the course and demonstrates practical implementation of classification techniques learned during the curriculum.

---
