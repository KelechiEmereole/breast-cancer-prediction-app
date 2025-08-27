## Breast Cancer Prediction App

**[Live Demo on Streamlit](https://breast-cancer-prediction-app-jyqgijsvktdxvt7t5s67te.streamlit.app/)**

### Project Overview

This project builds and deploys a Machine Learning web application that predicts the likelihood of breast cancer (benign or malignant). The app is designed using Streamlit and powered by a Logistic Regression model, carefully chosen after evaluating multiple classification techniques.

### STAR Breakdown

### Situation
Early detection of breast cancer is crucial for saving lives. Medical datasets often contain many features, and not all of them equally contribute to diagnosis. A robust ML model with streamlined features can aid in accurate, interpretable, and faster predictions.

### Task
My task was to:

* Analyze breast cancer dataset from UCI.

* Perform feature importance and reduce 32 features to the 13 most important ones.

* Build, compare, and evaluate multiple classification models.

* Deploy the best-performing model into a user-friendly web app using Streamlit.

### Action

* Applied feature importance to reduce to the top 13 impactful features for efficiency.

* Modeling (5 classification models):

* Logistic Regression → Accuracy = 97.36%, Recall = 95.23%, AUC = 0.9976 

* Decision Tree → Accuracy = 92.98%, Recall = 85.71%, AUC = 0.9146

* Random Forest → Accuracy = 96.49%, Recall = 90.47%, AUC = 0.9918

* Support Vector Classifier → Accuracy = 97.36%, Recall = 95.23%, AUC = 0.9953

* KNN → Accuracy = 93.85%, Recall = 85.71%, AUC = 0.9836

### Model Selection:
Logistic Regression was chosen because it achieved the highest accuracy (97.36%) and AUC (0.9976), while maintaining strong recall (95.23%). It is also simple, interpretable, and less prone to overfitting, making it ideal for medical decision-making.

### Deployment:
Designed a Streamlit app where users can input 13 tumor features (e.g., texture_worst, radius_se, symmetry_worst, etc.) and instantly get a prediction of Benign or Malignant.

### Result

Built a production-ready ML app with ~97% accuracy and near-perfect AUC.

Reduced features from 32 → 13, making the model faster and more interpretable.

Provided a reliable decision-support tool for healthcare research and diagnostics.

### Feature Importance

Top features like texture_worst, radius_se, and symmetry_worst strongly impact diagnosis.

Reducing to 13 features preserved accuracy while improving interpretability.

### Medical Application

The model is not a replacement for medical diagnosis but can be a support tool for clinicians.

High recall ensures fewer false negatives, critical in cancer detection.

### Future Improvements

* Expand dataset with more diverse samples.

* Explore deep learning models for comparison.

* Add SHAP/interpretability plots for explainable AI in healthcare.

### Tech Stack

Python (Pandas, NumPy, Scikit-learn)

Machine Learning Models: Logistic Regression, Decision Tree, Random Forest, SVC, KNN

Streamlit (for deployment)

Matplotlib / Seaborn (for visualization)

Joblib (for model persistence)

**Thank you for taking the time to read this report!**

**Please reach out for any updates.**

### Author
[Kelechi Emereole](https://github.com/KelechiEmereole)
