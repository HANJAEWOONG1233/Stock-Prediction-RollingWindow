# <img src="https://img.shields.io/badge/Stock--Prediction-4Hr%20Rolling%20%2B%20Factor%20Backtest-9cf?style=for-the-badge" alt="banner" />  
# Stock-Prediction-4-Hour Rolling Prediction & Factor-Based Backtesting

> This project predicts the midpoint \((\frac{high + low}{2})\) of 4-hour bars using **LightGBM**, then adjusts the prediction using a **factor** for long/short strategies.  
> Below is a **code structure** summary, **backtest results**, and placeholders for **graphs** in the README layout.

<br />

---

## 1. Introduction  🎯

- **Objective**  
  - Predict the midpoint \((high + low)/2\) of each 4-hour candle using a **LightGBM** model  
  - Implement a **long/short** strategy (predicted value up → go long, down → go short)  
  - Apply a **factor** (ranging from 0 to 0.3, with a 0.001 increment) to the predicted values, then evaluate **returns** and **MDD**  

- **Key Features**  
  - **Rolling Window** approach: train on a fixed window size (window_size) of past data, then predict the subsequent bar  
  - **TA-Lib** for automatic calculation of various technical indicators (moving averages, Bollinger Bands, MACD, etc.)  
  - **Backtesting** with an initial capital of \$1,000, a fixed 2 shares per trade, entry at open price and exit at close price, measuring returns and MDD  

<br />

---

## 2. Code Structure Explanation  🏗️

### A. Rolling Window Prediction  
1. **Data Ingestion**  
   - Load 4-hour CSV files  
   - Calculate multiple technical indicators using TA-Lib (moving averages, Bollinger, MACD, RSI, etc.)

2. **Feature Construction**  
   - Prediction target: \((high + low)/2\)  
   - Each indicator is **shifted(1)** so only past information is used for the prediction of the current bar

3. **LightGBM Training**  
   - **window_size** amount of historical data is used in a sliding manner  
   - For time \(i\), train on the previous window, then predict bar \(i\), and move the window forward

4. **Factor Application**  
   - If the predicted value is higher than open → go long → multiply the predicted value by \((1 - factor)\)  
   - If the predicted value is lower than open → go short → multiply by \((1 + factor)\)  
   - The factor ranges from 0 to 0.3 in increments of 0.001

5. **Result CSV Output**  
   - Outputs the predicted midpoint `predicted_mid`, along with columns for each factor `pred_adj_x.xx`

<br />

### B. Factor-Based Backtesting  
1. **Backtest Logic**  
   - Start with \$1,000 in capital, trade 2 shares (fixed) per signal  
   - If the new prediction is higher than the previous prediction → **enter long**, otherwise → **enter short**  
   - Enter at the **open** price, exit at the **close** price  

2. **Returns & Equity**  
   - Long trades: \(\frac{exitPrice - entryPrice}{entryPrice}\), short trades: \(\frac{entryPrice - exitPrice}{entryPrice}\)  
   - Profits (or losses) are reflected in the **current equity** after each trade  

3. **Maximum Drawdown (MDD)**  
   - Measures the largest drop from the highest equity point seen throughout the backtest  

4. **Factor Comparison**  
   - Runs a backtest on each factor column, from `pred_adj_0.000` to `pred_adj_0.300`  
   - Identifies the **best-performing** factor in terms of total return, and plots the equity/drawdown graph  

<br />

---

## 3. Hyperparameter Tuning  🔧

- Used **Grid Search** for multiple candidate parameters (learning rate, tree depth, etc.)  
- Adopted SMAPE (also RMSE or MAE possible) as the metric to minimize  
- **Empirical/experimental** approach rather than a purely theoretical optimum  

<br />

---

## 4. Final Backtest Results  📈

We tested four different assets (Tesla, Nvidia, Apple, and Bitcoin) on **4-hour** datasets.  
The **returns** and **MDD** are summarized below. You can insert **equity/drawdown** graphs under each result.

<br />

### A. Tesla (4H, 2018.01 ~ 2025.01)

- **Total Return**: **214.54%**  
- **MDD**: **18.72%**

![image](https://github.com/user-attachments/assets/466fdfe4-f5a1-4716-ab7e-abf6057c6cf5)

<br />

---

### B. Nvidia (4H, 2018.01 ~ 2025.01)

- **Total Return**: **84.83%**  
- **MDD**: **11.57%**

![image](https://github.com/user-attachments/assets/f543e7b8-45e2-4bdc-bd24-7f5c7a317323)

<br />

---

### C. Apple (4H, 2018.01 ~ 2025.01)

- **Total Return**: **22.23%**  
- **MDD**: **3.11%**

![image](https://github.com/user-attachments/assets/09909912-58d3-4943-ad2a-4945c082f7d4)

<br />

---

### D. Bitcoin (4H, 2022.07 ~ 2024.12)

- **Total Return**: **76.38%**  
- **MDD**: **15.29%**

![image](https://github.com/user-attachments/assets/7f61b178-1416-4a11-a8fa-4ac31cc90aed)

<br />

---

## 5. Usage Guide  📝

1. **Prepare Data**  
   - Gather 4-hour CSV files (must include time, open, high, low, close, volume)  
   - If using TA-Lib, install it along with any indicators needed (e.g. RSI, MACD, etc.)

2. **Run Rolling Prediction**  
   - Execute a **rolling window** regression to predict \((high + low)/2\)  
   - Generate columns for factors (0 to 0.3)

3. **Run Backtest**  
   - Evaluate **returns/drawdown** for each factor  
   - Identify the factor with the best performance and review the equity and drawdown charts  

<br />

---

## 6. Notes & Future Work  💡

- Consider **commission** and **slippage** to simulate real-world market conditions more closely  
- **Leverage** can be applied to amplify both returns and volatility  
- Expand the approach to other timeframes (e.g., **1H, Daily**) using the same logic  

<br />

---

## 7. License  📜

- Unless otherwise stated, all materials in this project adhere to the [MIT License](https://opensource.org/licenses/MIT).

<br />

---

## 8. Final Summary  🎉

- **Midpoint prediction** \((\frac{high + low}{2})\) combined with **factor adjustments** for long/short in 4-hour bars  
- Solid **cumulative returns** achieved across Tesla, Nvidia, Apple, and Bitcoin  
- Further **strategy refinement** possible through additional indicators, transaction fees, and leveraging  

<br />

> **Inquiries & Issues**  
> If you encounter any problems or have questions, please open an Issue.  
> Wishing you the best in trading and research!
