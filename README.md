# Disease Prediction Using Machine Learning

This project applies supervised machine learning algorithms to predict diseases based on symptoms. The goal is to assist early diagnosis by training models on symptom-based features to accurately classify diseases.

## Project Overview

Using a dataset composed of patient symptoms and corresponding diagnoses, multiple machine learning classifiers were evaluated to identify the most effective model for predicting disease. The models were trained and evaluated using performance metrics such as accuracy, precision, recall, and F1-score.

## Dataset

- **Features**: Binary or categorical symptom indicators
- **Target**: Disease category (multi-class)
- **Size**: The dataset originally contains 4988 rows and 133 columns but for more accurate results, duplicate rows have been 
  removed which reduces the size of the dataset to around 305 rows and 133 columns.

## Technologies Used

- Python
- Jupyter Notebook
- scikit-learn, NumPy, pandas
- matplotlib / seaborn (for visualizations)

## Key Features

- Cleaned and preprocessed a symptom-disease dataset
- Built and compared five classification models
- Evaluated models using standard metrics
- Visualized accuracy across classifiers for better interpretability

## 📊 Model Performance

| Classifier                      | Accuracy (%) |
|--------------------------------|--------------|
| Naïve Bayes (GNB)              | **87.13** ✅ |
| Random Forest                  | **87.13** ✅ |
| Support Vector Machine (SVM)   | **87.13** ✅ |
| K-Nearest Neighbors (KNN)      | 74.26        |
| Decision Tree                  | 73.27        |

**Conclusion**: Naïve Bayes, SVM, and Random Forest classifiers performed equally well, each achieving an accuracy of approximately 87.13%. Naïve Bayes is particularly promising due to its simplicity and speed on this dataset.

## How to Run

1. Clone the repository and open the notebook file.
2. Install required packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
Run the notebook cells in order to train models and view results.
