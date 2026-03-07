# Sales Forecasting — Fresh Analytics
### Predicting Item Demand Across Restaurants (2019–2021)

**Author:** Ajaykanna E  
**Type:** Capstone Project — Artifical Intelligence Engineering

---

## Project Overview

Fresh Analytics operates a network of restaurants and needs to accurately forecast daily item demand to optimize staffing, inventory, and supply chain decisions.

This project walks through the **complete data science workflow** — from raw data exploration to deploying both machine learning and deep learning forecasting models — covering three years of real sales data (2019–2021) and generating forecasts for 2022.

---

## Dataset

Three CSV files are used in this project:

| File | Description |
|------|-------------|
| `sales.csv` | Daily item-level sales records (item count + revenue) |
| `items.csv` | Item metadata — name, calories, unit cost |
| `resturants.csv` | Restaurant names and IDs |

After merging, the final dataset contains **109,600 rows** spanning January 2019 to December 2021.

---

## Project Structure

```
├── Sales_Forecasting_Complete_final.ipynb   # Main notebook
├── sales.csv                                # Sales data
├── items.csv                                # Item metadata
├── resturants.csv                           # Restaurant data
└── README.md
```

---

## Sections Covered

### Section 1 — Preliminary Analysis
- Imported and examined all three datasets
- Checked for missing values (none found)
- Detected outliers using the **IQR method** on `item_count` and `price`
- Merged all three datasets into a single enriched DataFrame (109,600 rows)

### Section 2 — Exploratory Data Analysis (EDA)
Extracted time features (year, month, quarter, day of week) and answered key business questions:

| Analysis | Key Finding |
|----------|-------------|
| Overall trend | Clear weekly cycles + gradual year-over-year growth |
| Day of week | **Friday** has highest sales; **Sunday** is the quietest |
| Monthly trend | Sales peak **May–August** (summer); dip in **January & December** |
| Quarterly | **Q2** is the strongest quarter; **Q4** is the weakest |
| Restaurant performance | **Bob's Diner** accounts for ~94% of all sales |
| Most popular item | **Strawberry Smoothy** — 236,000+ units sold |
| Revenue vs volume | Bob's Diner leads in both metrics |
| Premium item | **Fou Cher** has the most expensive item at $53.98 |

### Section 3 — Machine Learning Forecasting
Built and compared three models for daily item count prediction using time-based train/test split (train: 2019–mid 2021 / test: Jul–Dec 2021):

| Model | Approach | Result |
|-------|----------|--------|
| Linear Regression | Baseline | High RMSE — cannot capture non-linear patterns |
| Random Forest | Ensemble (200 trees) | ~4× lower RMSE than Linear Regression |
| XGBoost | Gradient Boosting (200 trees) | Comparable to Random Forest |

**Feature Importance:** `dayofweek` was the most influential feature (~0.53 importance score), followed by `dayofyear` — validating the patterns found in EDA.

**2022 Forecast:** The best-performing model was used to generate a full-year 2022 item count forecast.

### Section 4 — Deep Learning with LSTM
Built a two-layer LSTM network to forecast daily **sales revenue**:

- **Preprocessing:** MinMaxScaler normalization, 30-day sliding window sequences
- **Architecture:** LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16) → Dense(1)
- **Training:** Adam optimizer, EarlyStopping (patience=10), 10% validation split
- **Result:** **MAPE = 10.55%** → "Good" performance range

A second LSTM model trained on the **full series** (including seasonally-aware synthetic data) was used to generate a **3-month recursive forecast (Jan–Mar 2022)**.

---

## 📊 Model Performance Summary

| Model | Metric | Score |
|-------|--------|-------|
| Linear Regression | RMSE | High |
| Random Forest | RMSE | Low  |
| XGBoost | RMSE | Low  |
| LSTM | MAPE | **10.55% (Good)** |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn (LinearRegression, RandomForestRegressor) |
| Gradient Boosting | XGBoost |
| Deep Learning | TensorFlow / Keras (LSTM) |
| Preprocessing | MinMaxScaler (sklearn) |

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
   ```

3. **Place the data files** (`sales.csv`, `items.csv`, `resturants.csv`) in the same directory as the notebook.

4. **Run the notebook**
   ```bash
   jupyter notebook Sales_Forecasting_Complete_final.ipynb
   ```
   Run all cells from top to bottom.

---

## 💡 Key Business Insights

- **Bob's Diner** dominates the network — any operational decision based on aggregate data is effectively a decision about Bob's Diner.
- **Friday staffing and inventory** should always be higher — it consistently has the highest sales across all three years.
- **Q2 (April–June)** requires the most resources; plan promotions and bulk procurement accordingly.
- The LSTM and ensemble ML models are accurate enough for **operational planning** (staffing, inventory ordering, supplier negotiations).

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
