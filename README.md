

# 🩺 Diabetes Risk Prediction using Machine Learning

![Poster Snapshot]([Poster\Diabetes Prediction Model Poster.png](https://github.com/AviralVashistha/Diabetes_Prediction/blob/main/Poster/Diabetes%20Prediction%20Model%20Poster.png))

> 🏆 *Awarded Best Poster at AMSC, IIT Roorkee – Foundation Day 2025*

This project aims to predict diabetes risk using multiple machine learning models and improve clinical interpretability using Explainable AI. Leveraging a dataset of 10,000+ patient records, we implemented models such as Random Forest, XGBoost, SVM, and MLP, along with LIME (Local Interpretable Model-Agnostic Explanations) to ensure transparency in predictions.

---

## 📁 Dataset

The dataset used in this project is based on real patient records with the following attributes:

| Feature                      | Description                                  |
| ---------------------------- | -------------------------------------------- |
| `Pregnancies`              | Number of pregnancies                        |
| `Glucose`                  | Plasma glucose concentration                 |
| `BloodPressure`            | Diastolic blood pressure (mm Hg)             |
| `SkinThickness`            | Triceps skin fold thickness (mm)             |
| `Insulin`                  | 2-Hour serum insulin (mu U/ml)               |
| `BMI`                      | Body Mass Index                              |
| `DiabetesPedigreeFunction` | Family history-based function score          |
| `Age`                      | Age in years                                 |
| `Outcome`                  | Class label (0 = Non-diabetic, 1 = Diabetic) |

---

## 📊 Exploratory Data Analysis

* Checked for **missing values** using `.isnull().sum()`
* Generated **correlation heatmaps** with Seaborn
* Analyzed **class imbalance** using count plots
* Identified key correlations between glucose, HbA1c, BMI, and diabetes outcome

---

## 🧠 Machine Learning Models Implemented

* ✅ **Random Forest** (Best performer)
* 🌲 **Tuned Random Forest** (via GridSearchCV)
* ⚙️ **XGBoost**
* 🧭 **Support Vector Machine (SVM)**
* 🔍 **Logistic Regression**
* 🤖 **Multi-layer Perceptron (MLP)** *(in extended version)*

Each model was trained, tested, and evaluated based on performance metrics.

---

## 📈 Evaluation Metrics

| Metric               | Description                                |
| -------------------- | ------------------------------------------ |
| `Accuracy`         | Overall correctness of model               |
| `Precision`        | TP / (TP + FP)                             |
| `Recall`           | TP / (TP + FN)                             |
| `F1-Score`         | Harmonic mean of precision & recall        |
| `ROC Curve & AUC`  | Evaluated probability-based classification |
| `RMSE / MAE / MBE` | Error evaluation for regression comparison |

> Random Forest achieved **lowest RMSE (0.166283)** and  **MAE (0.02765)** , making it the most reliable in minimizing error.

---

## 📌 Explainable AI with LIME

To enhance interpretability, LIME was used to analyze feature importance for individual predictions.

🔍  **Key Insights from LIME** :

* `Blood Glucose Level` is the most influential predictor.
* `HbA1c Level` negatively impacts diabetes risk (reducing risk).
* Other features like BMI, Age, and Hypertension have moderate influence.

---

## 🔧 Hyperparameter Tuning

Performed using `GridSearchCV` on Random Forest with:

* 3-fold cross-validation
* Accuracy as scoring metric

---

## 📊 Visualizations Included

* ✅ Correlation heatmaps
* ✅ Confusion matrices
* ✅ ROC-AUC curves
* ✅ Model accuracy comparisons
* ✅ LIME-based feature importance bar plots

---

## 🚀 How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/diabetes-risk-prediction.git
   cd diabetes-risk-prediction
   ```
2. **Install required libraries**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the prediction script**
   ```bash
   python diabetes_prediction.py
   ```

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lime
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lime
```

---

## 🔬 Key Outcomes

* Random Forest performed the best across  **accuracy** ,  **error** , and  **interpretability** .
* Glucose and HbA1c levels were the  **most significant predictors** .
* LIME successfully enhanced  **model transparency** , aiding medical understanding.

---

## 📍 Future Scope

* Reduce **False Negatives** via advanced tuning
* Incorporate **real-time risk prediction system** for clinical use
* Explore deep learning models for high-dimensional medical data

---

## 📃 License

This project is licensed under the [MIT License](https://chatgpt.com/c/LICENSE).

---

## 🙌 Acknowledgments

Special thanks to:

* SRM Institute of Science & Technology
* AMSC Department, IIT Roorkee
* Mentors and Faculty
* Kaggle Community for dataset resources

---

## 👥 Authors

* **Aviral Vashistha**
* **Bhavya Bharadwaj**

---
