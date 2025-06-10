# Transformer Stock Predictor - Implementation Plan

This document outlines the planned enhancements for the Transformer Stock Predictor application, focusing on advanced features for data handling, prediction, and user experience.

---

## **Phase 1: Core Enhancements & Data Management**

### **1. Expand Stock List & Add Stock Metadata**
- **Goal:** Include a broader range of stocks and display relevant company details.
- **Details:**
    - Expand `STOCK_DB` in `app.py` to include more NSE stocks (e.g., NIFTY 50, NIFTY Next 50).
    - Fetch and display basic stock metadata (e.g., sector, industry, market cap) in the "Price Analysis" tab. This data will be fetched from `yfinance`.
- **Completion Criteria:** Broader stock selection available, and new metadata displayed for the selected stock.

### **2. Implement Flexible Time Filters**
- **Goal:** Allow users to view data and predictions across various time horizons.
- **Details:**
    - Add a Streamlit selectbox or radio buttons for time filtering: `1D`, `1W`, `1M`, `YTD`, `1Y`, `5Y`, `Max`.
    - Dynamically adjust the `start_date` for data fetching based on the selected filter.
- **Completion Criteria:** Time filters are present and correctly adjust the displayed data range.

---

## **Phase 2: Advanced Prediction Features**

### **3. Implement Prediction Range / Confidence Intervals**
- **Goal:** Provide a more nuanced understanding of prediction uncertainty.
- **Details:**
    - Modify the `transformer_model.py` to support Monte Carlo Dropout (enable dropout during inference).
    - Run multiple forward passes (e.g., 50-100 times) during prediction to get a distribution of outputs.
    - Calculate and display a prediction range (e.g., 95% confidence interval) on the "Predictions" chart.
- **Completion Criteria:** Prediction chart shows a shaded confidence interval around the point prediction.

### **4. Add Direction Probability**
- **Goal:** Quantify the likelihood of upward, downward, or flat movement.
- **Details:**
    - Based on the Monte Carlo predictions, calculate the probability of the next day's (or chosen horizon's) price being higher, lower, or within a small range of the current price.
    - Display these probabilities as percentages (e.g., in metric cards or a pie chart).
- **Completion Criteria:** Direction probabilities (Up/Down/Flat) are displayed in the "Predictions" tab.

### **5. Implement Multi-Horizon Predictions**
- **Goal:** Predict price movements for different future timeframes.
- **Details:**
    - Extend the prediction logic to generate predictions for the next day, next week (e.g., 5 trading days), and next month (e.g., 20 trading days).
    - Display these multi-horizon predictions on the chart or in separate sections.
- **Completion Criteria:** Predictions are available for multiple future horizons.

---

## **Phase 3: Automation & Deeper Insights**

### **6. Implement Periodic Training (on Working Days)**
- **Goal:** Keep the model updated with the latest market data.
- **Details:**
    - Create a separate script or modify `train.py` to be callable for scheduled retraining.
    - **Note:** Actual scheduling (e.g., cron job, GitHub Actions) is outside the immediate scope of `app.py` but will be outlined. The primary implementation will be the retraining logic.
    - Integrate `training_history.json` to log each automated training session.
- **Completion Criteria:** A manual trigger (or placeholder for automated trigger) exists to retrain the model with new data.

### **7. Enhance Explainability with SHAP/LIME (Consider)**
- **Goal:** Provide more robust reasons for predictions.
- **Details:**
    - Integrate SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to interpret model predictions.
    - Visualize feature importance for specific predictions.
- **Completion Criteria:** A dedicated section shows SHAP/LIME explanations for selected predictions.

---

## **Phase 4: Advanced Analysis & UX Refinements**

### **8. Backtesting Functionality**
- **Goal:** Allow users to simulate trading strategies and evaluate performance.
- **Details:**
    - Implement a backtesting engine that uses historical predictions to simulate buy/sell signals.
    - Display key backtesting metrics: CAGR, Sharpe Ratio, Max Drawdown, number of trades, profit/loss.
- **Completion Criteria:** A "Backtesting" tab is available with configurable strategy and performance metrics.

### **9. Model Comparison Interface**
- **Goal:** Enable side-by-side comparison of different model performances.
- **Details:**
    - Allow loading/selection of different trained models (e.g., Transformer, LSTM, ARIMA).
    - Display comparative charts and metrics for each model.
- **Completion Criteria:** Users can select and compare different models.

--- 