# 🛡️ Vendor Invoice Intelligence System

An end-to-end Machine Learning pipeline designed to predict freight costs and detect potentially anomalous or risky vendor invoices using regression and classification algorithms.

## ✨ Features
* **Automated Preprocessing:** Handles missing values and fixes negative edge-case inputs gracefully.
* **Smart Feature Engineering:** Builds business-relevant features like `total_cost`, simulated `freight_cost`, and aggregated vendor histories.
* **Dual ML Models:** 
  * `RandomForestRegressor` to estimate expected freight costs.
  * `RandomForestClassifier` to flag risky transactions based on anomalous cost margins.
* **Interactive UI:** A clean Streamlit dashboard to predict outcomes on the fly.

## 💻 Tech Stack
* Python 3.9+
* Pandas & NumPy (Data Manipulation)
* Scikit-Learn (Machine Learning pipeline)
* Streamlit (Frontend UI)

## 🚀 Setup Instructions

1. **Clone the repository and navigate into it.**
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
