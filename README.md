# USD/SGD Exchange Rate Prediction with ML & NLP

This repository contains a comprehensive project aimed at predicting the USD/SGD exchange rate using a combination of **financial indicators**, **sentiment analysis**, and **machine learning models**. The approach integrates data preprocessing, NLP sentiment scoring, Random Forest regression, LSTM networks, and Particle Filters to provide robust short- and medium-term forecasts.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Data Sources](#data-sources)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Data Cleaning](#1-data-cleaning)
  - [2. Correlation Testing](#2-correlation-testing)
  - [3. Model Evaluation](#3-model-evaluation)
  - [4. Feature Selection](#4-feature-selection)
  - [5. Sentiment Analysis with NLP](#5-sentiment-analysis-with-nlp)
  - [6. LSTM Modeling](#6-lstm-modeling)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)


---

## Project Overview

With the growing importance of US–Singapore trade relations, forecasting the USD/SGD exchange rate has significant implications for investors, financial institutions, and policymakers. This project explores how combining **macroeconomic factors**, **market indices**, **commodity prices**, and **news sentiment** can yield predictive insights into exchange rate movements.

The main goals of this project are:
1. **Data Integration and Cleaning**: Aggregate financial data (exchange rates, commodity prices, market indices) and public news sentiment to form a consolidated dataset.
2. **Feature Engineering**: Use statistical methods (e.g., PCA, feature importance) to select the most impactful predictors.
3. **Machine Learning Model Development**: Implement Random Forest and LSTM models for time-series forecasting.
4. **Sentiment Analysis**: Incorporate NLP-based sentiment scores (VADER, FinBERT) to assess media impact on exchange rate movements.
5. **Model Evaluation**: Compare predictions with actual exchange rates using MAE, RMSE, R^2, and other metrics.

---

## Key Features

- **Data Preprocessing**: Handles missing/infinite values, scales/normalizes features, and calculates moving averages/log returns.
- **Sentiment Analysis**: Leverages VADER and potentially FinBERT or custom lexicons to extract positive/negative sentiment from news text.
- **Random Forest Regression**: Provides feature importance rankings and baseline regression accuracy.
- **LSTM Model with Attention Mechanism**: Captures both short- and long-term dependencies in time-series. The attention layers help the model focus on critical parts of the sequence.
- **Particle Filter**: Applies a rolling window (T-3) approach to forecast exchange rate log returns, allowing dynamic adaptation to recent market changes.

---

## Data Sources

1. **World Bank / Yahoo Finance / FRED**: Main source for macroeconomic indicators such as GDP, interest rates, inflation data, and daily exchange rates for USD/EUR/JPY/CNY.
2. **Market Indices**: S&P 500 (US) and STI (Singapore), Gold Price, Dollar Index (DXI). Additional data sourced from Yahoo Finance.
3. **Public News Data**: Over 200k+ headlines or articles related to the USD and SGD, labeled with sentiment. Source: [Kaggle](https://www.kaggle.com/)
4. **Proprietary CSV/Parquet Files**: 
   - `selected_data.parquet` (Due to the size, the data can be find in kaggle)
   - `Dataset.xlsx`
   - `Dataset_1990-2020.xlsx`
   - `Processed_Data.csv`
   - More details in `Models` and `Data`.

---

## Dependencies

Below is a non-exhaustive list of required Python libraries:

- **Python 3.7+**
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [nltk](https://www.nltk.org/) (including `vader_lexicon`)
- [tensorflow](https://www.tensorflow.org/) / [keras](https://keras.io/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/) (optional for additional visualization)
- [PyPDF2](https://pypi.org/project/PyPDF2/) (if needed for PDF extraction in certain scripts)

Make sure you install all dependencies via `pip install -r requirements.txt` or manually.

---
## Project Structure

This repository is organized as follows:

### Data
- **Dataset.xlsx**: Initial raw dataset.
- **Dataset_1990-2020.xlsx**: Dataset covering the years 1990 to 2020.
- **Processed_Data.csv**: Cleaned and processed data ready for analysis.

### Model
- **1 Data Cleaning.ipynb**: Notebook for initial data cleaning steps.
- **2 Correlation Test.ipynb**: Analysis notebook for testing data correlations.
- **3 Other Model Check.ipynb**: Notebook for evaluating additional models.
- **4 Feature Choose.ipynb**: Feature selection notebook.
- **5 NLP.ipynb**: Notebook for Natural Language Processing tasks.
- **6 LSTM.ipynb**: Notebook for LSTM model development.

### Report
- **Code Output.pdf**: PDF containing outputs from code execution.
- **Report.pdf**: Detailed project report discussing findings and methodologies.

### Others
- **LICENSE.md**: License details for the project.
- **README.md**: Documentation for navigating and understanding the project.


---

## Usage

Ensure that your Python environment has all dependencies installed (`pip install -r requirements.txt`). Then follow the notebooks in sequence:

### 1. Data Cleaning
- Load and preprocess raw financial data from various sources.
- Handle missing values and anomalies.
- Compute technical indicators like moving averages and log returns.

### 2. Correlation Testing
- Perform statistical correlation tests (e.g., Pearson's correlation, Spearman’s rank) to determine relationships among variables.
- Results help narrow down potential predictors.

### 3. Model Evaluation
- Evaluate traditional regression and ensemble models (Linear Regression, Ridge, Lasso, ElasticNet, Random Forest).
- Establish baseline performance metrics (RMSE, MAE, R²).

### 4. Feature Selection
- Use Variance Inflation Factor (VIF) to detect multicollinearity among independent variables.
- Apply Permutation Importance to identify the most impactful predictors for modeling.
- Narrow down optimal feature sets based on these analyses.

### 5. Sentiment Analysis with NLP
- Conduct text analysis using VADER Sentiment Analyzer.
- Calculate sentiment scores and derive an "impact score" to quantify the influence of financial news on exchange rates.
- Integrate sentiment scores into your feature set.
- More detial process on Xiao, Q., & Ihnaini, B. (2023). *Stock trend prediction using sentiment analysis*.

### 6. LSTM Modeling
This notebook focuses specifically on building and fine-tuning an LSTM-based predictive model. Key steps include:

- **Data Preparation**:  
  Prepare sequential datasets suitable for LSTM input, normalizing features and targets separately.
  
- **Model Architecture**:  
  Develop a multi-layer Bi-directional LSTM with attention layers:
  - Input layer configured for sequential data.
  - Multiple stacked Bi-directional LSTM layers with Dropout (to prevent overfitting) and Batch Normalization (for stability and faster training).
  - Attention mechanism to improve sequence modeling by focusing on relevant temporal features.
  - Final Dense output layer producing exchange rate forecasts.

- **Training and Validation**:  
  Train with an adaptive optimizer (Adam), monitoring validation loss to trigger early stopping and prevent overfitting.

- **Evaluation**:  
  Analyze performance using various metrics including RMSE, MAE, and R² scores.  
  Compare LSTM results against baseline predictions and assess the improvement.

- **Visualization**:  
  Plot predicted versus actual exchange rates over time to visualize model accuracy and reliability clearly.

- **Interpretation**:  
  Discuss attention weights and feature importance, highlighting how certain features contribute significantly to predictive accuracy.



---

## Results

- **Random Forest** typically shows high R^2 (≈0.99) with low MSE, indicating strong predictive capability for the scaled data.
- **LSTM with Attention** outperforms a simple baseline (T-3 shift) with a lower MAE, demonstrating its ability to capture complex temporal dependencies.
- **Particle Filter** helps illustrate market volatility in a rolling context and adapts predictions in near real-time.

### Visualizations

![Random Forest Results](https://github.com/Haonan-100/US-Dollar-to-Singapore-Dollar-Exchange-Rate-Prediction/blob/main/photo/01.png)
![LSTM Results](https://github.com/Haonan-100/US-Dollar-to-Singapore-Dollar-Exchange-Rate-Prediction/blob/main/photo/02.png)

---

## Future Work

1. **Expanded Feature Set**: Incorporate additional macroeconomic indicators, real-time streaming data, or social media sentiment (e.g., Twitter).
2. **Hybrid Architectures**: Combine more advanced NLP pipelines (FinBERT, domain-specific dictionaries) with extended forecasting models like Transformers.
3. **Real-Time Deployment**: Develop a pipeline to automatically ingest live data, update LSTM weights periodically, and produce rolling forecasts.

---

## References

- Xiao, Q., & Ihnaini, B. (2023). *Stock trend prediction using sentiment analysis*. [PeerJ Computer Science](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10403218/).
- [Random Forest Regression Documentation](https://apple.github.io/turicreate/docs/userguide/supervised-learning/random_forest_regression.html)
- Jie Du et al. (2020). *Power Load Forecasting Using BiLSTM-Attention*. IOP Conf. Ser.: Earth Environ. Sci. 440 03211.

---

**Thank you for visiting this project.** If you have any questions or suggestions, feel free to open an issue or submit a pull request.


