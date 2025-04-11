# USD/SGD Exchange Rate Prediction with ML & NLP

This repository contains a comprehensive project aimed at predicting the USD/SGD exchange rate using a combination of **financial indicators**, **sentiment analysis**, and **machine learning models**. The approach integrates data preprocessing, NLP sentiment scoring, Random Forest regression, LSTM networks, and Particle Filters to provide robust short- and medium-term forecasts.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Data Sources](#data-sources)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Sentiment Analysis](#2-sentiment-analysis)
  - [3. Random Forest Regression](#3-random-forest-regression)
  - [4. LSTM Model with Attention](#4-lstm-model-with-attention)
  - [5. Particle Filter Predictions](#5-particle-filter-predictions)
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
2. **Market Indices**: S&P 500 (US) and STI (Singapore), Gold Price, Dollar Index (DXI).
3. **Public News Data**: Over 200k+ headlines or articles related to the USD and SGD, labeled with sentiment.
4. **Proprietary CSV/Parquet Files**: 
   - `selected_data.parquet`
   - `Database2.csv`
   - `non_zero_impact_score_examples.csv`
   - More details in `Total.py`.

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
.
├── Total.py                              # Main script integrating data prep, sentiment analysis, RF, LSTM, Particle Filter
├── Report.pdf                            # Project report with theoretical background and results discussion
├── selected_data.parquet                 # Example input data (if provided)
├── Database2.csv                         # Another dataset for model training
├── non_zero_impact_score_examples.csv    # Sample sentiment-labeled output data
├── requirements.txt                      # Optional: list of dependencies
└── README.md                             # Project documentation

---

## Usage

### 1. Data Preparation
- Ensure your data files (e.g., `selected_data.parquet` and `Database2.csv`) are in the correct paths specified inside `Total.py`.
- The code handles feature creation such as rolling means (`ExchangeRate_Short_MA`, `ExchangeRate_Long_MA`) and log returns.

### 2. Sentiment Analysis
- The script uses **VADER** (from NLTK) and can incorporate finance-specific keywords or external models like **FinBERT**.
- Sentiment scores are combined with keyword context to compute an `impact_score` for USD or SGD.

### 3. Random Forest Regression
- A `RandomForestRegressor` is trained on scaled features to predict scaled exchange rates.
- Feature importance metrics provide insight into which economic indicators matter most (e.g., DXI, STI, GoldPrice).

### 4. LSTM Model with Attention
- Constructs a multi-layer **BiLSTM** with dropout, batch normalization, and an attention mechanism.
- Trained using a rolling-window (T-3) approach to capture recent market trends.
- Evaluated on metrics such as MSE, RMSE, MAE, and R^2.

### 5. Particle Filter Predictions
- Uses predicted LSTM outputs combined with a particle filter to track log-return dynamics over time.
- Provides a visualization of how forecasted trajectories compare to actual values, highlighting model confidence.

---

## Results

- **Random Forest** typically shows high R^2 (≈0.99) with low MSE, indicating strong predictive capability for the scaled data.
- **LSTM with Attention** outperforms a simple baseline (T-3 shift) with a lower MAE, demonstrating its ability to capture complex temporal dependencies.
- **Particle Filter** helps illustrate market volatility in a rolling context and adapts predictions in near real-time.

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


