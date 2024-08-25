# MLOPS-Project

This project showcases three machine learning models:

1. **Mushroom Edibility Classification:** Predict whether a mushroom is edible or poisonous based on its features.
2. **Residential Price Regression:** Predict residential property prices using various features.
3. **Anomaly Detection in Transactions:** Identify anomalous transactions in expense records to detect potential fraud or errors.

## Model 1: Mushroom Edibility Classification
### **Deployed Web Application Link: https://mlops-project-8dpn.onrender.com**

### Description

This model classifies mushrooms as edible or poisonous based on their features using a random forest algorithm.

### File structure
Michelle
- data
    - 02_mushroom_species_data.csv
- model
    - label_encoders.pkl
    - mushroom_pipeline.pkl
- notebook
    - IT3385_IndivNotebook_221547F_Michelle.ipynb
- static
    - style.css
- templates
    - index.html
    - result.html
- mushroom.py
- requirements.txt

## Model 2: Residential Price Regression
### **Deployed Web Application Link: https://mlops-project-kyw7.onrender.com/**

### Description

This model forecasts the residential price using a catboost regression model.

### File structure
Kynan
- data
    - 01_residential_data.xlsx
- model
    - hdb_resale_pipeline.pkl
- notebook
    - MLOP 220149D assignment.ipynb
- templates
    - index.html
- app.py
- requirements.txt

## Model 3: Anomaly Detection in Transactions
### **Deployed Web Application Link: https://mlops-project-8dpn.onrender.com**

### Description

This model identifies anomalous transaction records by using multiple features. 

### File structure
Jo-lynn
- data
    - 03_transaction_records.csv
- model
    - knn_pipeline.pkl
- notebook
    - Assignment.ipynb
- static
    - style.css
- templates
    - home.html
- app.py
- requirements.txt
