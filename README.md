# 🇧🇷 Olist E-Commerce Logistics Optimization

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![SQL](https://img.shields.io/badge/PostgreSQL-DB-blue)](https://www.postgresql.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-green)](https://scikit-learn.org/)
[![Visualization](https://img.shields.io/badge/Dashboard-Chart.js-orange)](https://www.chartjs.org/)

## 📋 Project Overview
This project analyzes **100k+ real e-commerce orders** from Olist (a Brazilian marketplace) to optimize logistics and predict delivery delays.

By building an **End-to-End Data Pipeline**, I extracted raw data, stored it in a PostgreSQL database, analyzed it using SQL & Python, and built a Machine Learning model to identify the key drivers of late deliveries.

### 🎯 Key Objectives
* **Data Engineering:** Build an ETL pipeline to load raw CSVs into a relational database.
* **Analytics:** Use SQL to find revenue trends, top sellers, and product performance.
* **Machine Learning:** Train a Random Forest Regressor to predict delivery time.
* **Explainability:** Use SHAP values to understand *why* orders are delayed (Geography vs. Seasonality).

---

## 🏗️ Architecture & File Structure

This repository follows a complete data lifecycle:

| File | Purpose | Description |
| :--- | :--- | :--- |
| **`setup.py`** | **ETL Pipeline** | Automated script that reads raw CSV data, cleans headers, and loads it into **PostgreSQL**. |
| **`olist.sql`** | **Data Modeling** | SQL scripts to clean data, create relations (Primary/Foreign Keys), and build the `analytics_master` view. |
| **`output.py`** | **Data Export** | Connects to the DB, queries the master view, and exports a clean dataset for the dashboard. |
| **`olist_ml.ipynb`** | **Machine Learning** | EDA, Feature Engineering, and Random Forest training. Includes SHAP analysis for model explainability. |
| **`olist.html`** | **Frontend** | A standalone HTML/JS Dashboard using Chart.js to visualize the insights interactively. |

---

## 📊 Key Insights

### 1. Delivery Time Drivers
Using **SHAP (SHapley Additive exPlanations)**, we discovered that **Geography** is the strongest predictor of delivery time, followed significantly by **Seller processing time**.

### 2. Business Performance
* **Revenue Trends:** Validated peak sales periods (e.g., Black Friday) using SQL time-series analysis.
* **Top Categories:** Ranked product categories by revenue and order volume.

---

## 🛠️ Technology Stack

* **Database:** PostgreSQL (Localhost)
* **ETL & Scripting:** Python (`pandas`, `sqlalchemy`, `psycopg2`)
* **Machine Learning:** Scikit-Learn (Random Forest), SHAP
* **Visualization:** Matplotlib, Seaborn, Chart.js (HTML5)

---

## 🚀 How to Run This Project

### Prerequisites
* Python 3.x
* PostgreSQL installed locally
* The Olist Dataset (Kaggle) placed in a `data/` folder.

### Step 1: Database Setup (ETL)
Run the setup script to load data into your local Postgres instance.
*(Note: Ensure you have created a database named `olist` first)*
```bash
python setup.py
