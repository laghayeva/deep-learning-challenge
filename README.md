# Deep Learning Challenge: Alphabet Soup Charity Success Predictor

## 📌 Overview

This project uses deep learning to build a binary classification model that predicts whether applicants to the nonprofit Alphabet Soup will be successful if funded. The goal is to help Alphabet Soup make smarter funding decisions using a neural network trained on historical application data.

---

## 📂 Dataset

The dataset includes over 34,000 records with the following fields:

- **APPLICATION_TYPE**
- **AFFILIATION**
- **CLASSIFICATION**
- **USE_CASE**
- **ORGANIZATION**
- **INCOME_AMT**
- **SPECIAL_CONSIDERATIONS**
- **ASK_AMT**
- **IS_SUCCESSFUL** (target)

---

## 🔄 Process

### Step 1: Preprocess the Data
- Dropped unnecessary ID columns: `EIN` and `NAME`
- Combined low-frequency categories into `"Other"` using cutoff thresholds
- Applied one-hot encoding with `pd.get_dummies()`
- Split data into training and testing sets
- Scaled features using `StandardScaler`

### Step 2: Build and Evaluate Neural Network
- Created a Sequential Keras model
- Used two hidden layers (80 and 30 nodes)
- Trained for 100 epochs
- Evaluated model performance
- Saved model as `AlphabetSoupCharity.h5`

### Step 3: Optimize the Model
- Created new notebook: `AlphabetSoupCharity_Optimization.ipynb`
- Increased number of hidden layers and neurons
- Adjusted rare-category grouping thresholds
- Trained with more epochs (150)
- Saved model as `AlphabetSoupCharity_Optimization.h5`

---

## 🧠 Technologies Used

- Python 3.10
- Pandas & NumPy
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook (VSCode)

---

## 📁 Files

- `AlphabetSoupCharity.ipynb` – Initial model notebook
- `AlphabetSoupCharity.h5` – Base model file
- `AlphabetSoupCharity_Optimization.ipynb` – Optimized model notebook
- `AlphabetSoupCharity_Optimization.h5` – Optimized model file

---

## 📝 Model Report

### Overview of the Analysis

The goal of this analysis was to build a binary classification model that predicts whether applicants for funding from the nonprofit Alphabet Soup will be successful. By analyzing historical application data, the model aims to support smarter funding decisions and increase the effectiveness of future grants.

---

### Results

#### Data Preprocessing

- **Target variable**:
  `IS_SUCCESSFUL` — indicates whether funding led to a successful outcome (1 = yes, 0 = no)

- **Features used**:
  All other columns after dropping `EIN` and `NAME`, and converting categorical features using one-hot encoding

- **Variables removed**:
  `EIN` and `NAME` — these were unique IDs and not useful for prediction

#### Compiling, Training, and Evaluating the Model

- **Model architecture**:
  - Input layer: Number of features = `X_train.shape[1]`
  - Hidden layers:
    - First: 80 neurons, ReLU activation
    - Second: 30 neurons, ReLU activation
  - Output layer: 1 neuron, sigmoid activation

- **Performance**:
  - Loss: ~0.59
  - Accuracy: ~73.1%

- **Training**:
  - 100 epochs using Adam optimizer and binary crossentropy loss

---

#### Optimization Attempts

In a second notebook (`AlphabetSoupCharity_Optimization.ipynb`), the following optimizations were made:

- Increased hidden layer count from 2 to 3
- Increased neurons to 128, 64, and 32
- Increased epoch count from 100 to 150
- Adjusted category grouping thresholds

**Final optimized model performance**:
- Accuracy: ~73.1%
- Slightly below the 75% target, but improved from baseline

---

### Summary

The final model achieved an accuracy of 73.1%, falling slightly short of the 75% benchmark. However, the model demonstrates a strong foundation and shows improvement through multiple optimization strategies.

### Recommendation for Future Improvements

For further improvement, consider testing:
- **Random Forest** or **XGBoost** for easier interpretability and performance
- Dropout layers or regularization
- Oversampling techniques like SMOTE to balance classes

---

## 👩‍💻 Author

This project was completed as part of the Columbia Engineering Data Analytics Bootcamp — Module 21 Challenge.
