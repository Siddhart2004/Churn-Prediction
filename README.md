# Customer Churn Prediction 

Predict which telecom customers are likely to churn using advanced machine learning techniques.

---

##  Project Overview

Customer churn is a major concern for subscription-based businesses. Reducing churn leads to improved customer lifetime value and stable revenue. In this project, I developed an end-to-end ML pipeline to predict churn and identify the key drivers behind it.

---

##  Tech Stack

- **Languages & Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn  
- **ML Frameworks:** Scikit‑learn (Random Forest, Logistic Regression), SMOTE (imbalanced-learn)  
- **Notebook Environment:** Jupyter Notebook

---

##  Dataset

The dataset includes telecom customer information such as:

- Customer demographics (age, gender)
- Service usage (total/data usage metrics)
- Billing details (monthly/total charges)
- Contract type and payment method
- Churn label (yes/no)

> Combines several CSV files: customer info, demographics, usage, and services.

---

##  Methodology

1. **Exploratory Data Analysis**  
   - Visualized churn distribution across features such as tenure, contract type, monthly charges  
   - Uncovered patterns: month-to-month contracts, high charges, senior citizens, and low satisfaction → higher churn

2. **Data Cleaning & Preprocessing**  
   - Handled missing values and inconsistent formats (e.g. some numeric fields stored as strings)  
   - Removed redundant features based on correlation  
   - Encoded categorical variables and standardized numeric data  

3. **Balancing the Dataset**  
   - Used **SMOTE** to address class imbalance before training  

4. **Model Training & Evaluation**  
   - Split data into training and test sets (stratified sampling)  
   - Built models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost  
   - Tuned hyperparameters using GridSearchCV  
   - Evaluated models via Accuracy, Precision, Recall, F1, ROC-AUC  
   - Best performance: **Random Forest & Logistic Regression (94% accuracy)**

5. **Ensemble Model**  
   - Combined Logistic Regression and XGBoost predictions (averaging probabilities)  
   - Improved Recall to ~93% using decision threshold (0.4)

---

##  Results & Insights

- Achieved **94% accuracy**, **macrof1 ~0.94**, **macro-recall ~0.93**  
- Important churn predictors: contract type, monthly charges, tenure, satisfaction score, payment method  
- Senior citizens and month-to-month plans have significantly higher churn rates

---

## 📚 How to Run

```bash
git clone https://github.com/Siddhart2004/Churn-Prediction.git
cd Churn-Prediction

# Set up Python environment
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Run the analysis
jupyter notebook Customer_Churn_Prediction.ipynb
