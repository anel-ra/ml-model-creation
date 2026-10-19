# Product Category Predictor 🚀

This project utilizes Machine Learning to automatically classify products into their respective categories based on the product title (`Product Title`) and the title length (`Title Length`).

The classification model is built using a `scikit-learn` cjevovod (`Pipeline`) that seamlessly combines natural language processing (text data) and numerical feature scaling.

## 📋 Project Features
- **Data Preprocessing & Cleaning**: Automatic text cleaning, removing hidden spaces (`strip`), converting to lowercase (`lower`), and category standardization (e.g., merging duplicate variants into unified classes like `cpus`, `fridges`, `mobile phones`).
- **Advanced Column Transformer**: 
  - `TfidfVectorizer` for extracting text features from product titles.
  - `MinMaxScaler` for scaling the length of the product titles.
- **Multi-Model Evaluation**: The training script compares multiple classifiers, including Logistic Regression, Naive Bayes, Decision Tree, Random Forest, and Support Vector Machine (LinearSVC).
- **Real-Time Prediction Interface**: A CLI script that allows users to input custom product titles in real-time and immediately receive the predicted category.

## 🛠️ Technologies & Libraries
The project is built using Python and the following core data science libraries:
- `pandas`
- `scikit-learn`
- `joblib`

## 📁 Project Structure
```text
ml-model-creation/
│
├── data/
│   └── dataset.csv              # Source product dataset
│
├── model/
│   └── predict_category_model.pkl  # Saved (trained) best performing model
│
└── src/
    ├── train_model.py           # Script for data cleaning and model training
    └── predict_category.py      # Script for running live interactive predictions
```

## 🚀 How to Run the Project

### 1. Install Dependencies
Before running the scripts, make sure you have installed the required libraries:
```bash
pip install pandas scikit-learn joblib
```

### 2. Train the Model
To clean the raw data, resolve duplicate classes, evaluate the classifiers, and save the best model, run the training script:
```bash
python src/train_model.py
```
*Note: During execution, a detailed `classification_report` (Precision, Recall, F1-score) will be displayed in your terminal for each model. The **Support Vector Machine (LinearSVC)** yields optimal performance (~96% accuracy).*

### 3. Run Live Predictions
Once the model is successfully exported to the `model/` directory, you can start the interactive prediction script:
```bash
python src/predict_category.py
```

**Terminal Example:**
```text
Model loaded successfully!
Type 'exit' at any point to stop.

 Enter product title: Intel Core i7-13700K Gaming Processor
 Predicted category: ['cpus']
----------------------------------------
 Enter product title: Samsung Galaxy S24 Ultra 512GB
 Predicted category: ['mobile phones']
----------------------------------------
 Enter product title: exit
Exiting...
```

## 🧠 Model & Evaluation Insights
Models are evaluated on a stratified test set (20% of total data) to guarantee class balance. The integrated `ColumnTransformer` handles feature mapping natively. When running live predictions, the script automatically computes
 the length of your input string, ensuring the data schema matches the exact structure (`Product Title` and `Title Length`) required by the pipeline, completely avoiding shape and mismatch errors (`ValueError`).