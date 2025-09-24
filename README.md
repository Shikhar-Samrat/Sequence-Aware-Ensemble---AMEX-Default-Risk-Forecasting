# Sequence-Aware Ensemble for AMEX Default Risk Forecasting

This project presents a sophisticated, sequence-aware ensemble model for predicting credit default, developed for the **American Express - Default Prediction Kaggle competition**. The solution leverages extensive feature engineering and combines the strengths of a Gradient Boosting model (LightGBM) and a deep learning model (Transformer-based Neural Network) to accurately forecast customer default risk.

## 🏆 Competition & Objective

The goal of the competition was to predict the probability that a customer will default on their credit card balance in the future based on their monthly profiles. The dataset is a time series of aggregated customer profiles, and the evaluation metric is a weighted average of the normalized Gini coefficient and a custom default rate capture metric.

## ⚙️ Methodology & Pipeline

The core of this project is a multi-stage pipeline that processes the raw data, engineers a rich set of features, and trains an ensemble of models.

The pipeline is executed sequentially as follows:

1.  **Denoising (`S1_denoise.py`):** The initial step involves preprocessing and cleaning the raw time-series data to remove noise and handle missing values, preparing it for feature engineering.

2.  **Manual Feature Engineering (`S2_manual_feature.py`):** Domain-specific features are created. This includes calculating aggregations, differences, and other statistical measures on the customer data to capture key behavioral patterns.

3.  **Series-Based Feature Engineering (`S3_series_feature.py`):** This step focuses on the sequential nature of the data. Features are generated from the time-series profiles of each customer to capture trends, seasonality, and other temporal dependencies.

4.  **Feature Combination (`S4_feature_combined.py`):** The manually engineered and series-based features are combined into a final, comprehensive feature set for model training.

5.  **LightGBM Model Training (`S5_LGB_main.py`):** A LightGBM model is trained on the combined feature set. LGBM is highly efficient and effective for tabular data, capturing complex non-linear relationships.

6.  **Neural Network Model Training (`S6_NN_main.py`):** A custom Transformer-based Neural Network is trained. This model is specifically designed to be "sequence-aware," using attention mechanisms to effectively process the time-series data.

7.  **Ensembling (`S7_ensemble.py`):** The predictions from the LightGBM and Neural Network models are combined using a weighted average to produce the final default probability scores. This ensemble approach leverages the diverse strengths of both models to improve overall prediction accuracy and robustness.

## 🚀 How to Run the Code

### 1\. Prerequisites

Ensure you have Python 3.8+ and the required libraries installed. You can install all dependencies using pip:

```bash
pip install -r requirements.txt
```

### 2\. Data Setup

Download the competition data from the [Kaggle competition page](https://www.kaggle.com/competitions/amex-default-prediction/data) and place the CSV files in an `input` directory at the root of the project.

### 3\. Execution

The entire pipeline can be executed by running the `run.sh` script:

```bash
bash run.sh
```

This script will sequentially execute all steps from denoising to the final ensembling, generating the submission file.

## 🗂️ File Structure

  * `S1_denoise.py`: Data preprocessing and denoising.
  * `S2_manual_feature.py`: Manual feature creation.
  * `S3_series_feature.py`: Time-series feature engineering.
  * `S4_feature_combined.py`: Merges all features.
  * `S5_LGB_main.py`: Trains the LightGBM model.
  * `S6_NN_main.py`: Trains the Transformer NN model.
  * `S7_ensemble.py`: Ensembles the predictions from both models.
  * `model.py`: Contains the definition for the Transformer NN architecture.
  * `utils.py`: Utility functions used across the project.
  * `run.sh`: Main execution script.
  * `requirements.txt`: Project dependencies.

## 👨‍💻 Contributors

  * **Shikhar Samrat**

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
