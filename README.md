# 📊 Facebook vs. AdWords Marketing Campaign Analysis (A/B Testing)

## Overview

This repository presents a comprehensive A/B testing analysis comparing the performance of Facebook Ads and Google AdWords campaigns. Using real campaign data and robust data science workflows, we assess which platform yields higher conversions, better efficiency, and actionable business insights for optimizing marketing investments.

## Contents

- **marketing_campaign.csv** — Raw campaign data for both ad platforms (views, clicks, conversions, costs, rates).
- **Market campaigns A/B testing.ipynb** — Jupyter Notebook containing all data cleaning, exploratory data analysis, testing, modeling, and visualization steps.

## Project Objectives

- Evaluate and compare the effectiveness of Facebook and AdWords campaigns using key metrics: clicks, conversions, cost per conversion, CTR, and conversion rates.
- Perform statistical hypothesis testing to determine if differences are significant.
- Build predictive models to estimate conversions from campaign parameters.
- Draw actionable recommendations to inform marketing strategy.

## Key Steps in the Notebook

1. **Data Import & Cleaning**
   - Standardizes column names and formats numeric columns for analysis.
2. **Exploratory Data Analysis**
   - Visualizes campaign performance metrics with histograms, bar charts, and conversion category breakdowns.
3. **Statistical Analysis**
   - Computes correlations and performs t-tests to compare channel effectiveness.
4. **Predictive Modeling**
   - Implements linear regression to predict conversions from clicks.
   - Explores trends by week and month.
5. **Insights & Conclusions**
   - Synthesizes findings and presents clear business recommendations.

## How to Run

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Ritesh799/Market-campaigns-A-B-testing.git
   cd Market-campaigns-A-B-testing
   ```

2. **Open `Market campaigns A/B testing.ipynb`** in Jupyter Lab, Jupyter Notebook, or Google Colab.

3. **Run the notebook cells sequentially.**
   - Make sure the CSV file is in the same directory as the notebook.

4. **Review outputs:**
   - All visualizations, statistical tests, and conclusions are generated within the notebook.

## Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- statsmodels

Install requirements using pip:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
```

## Results Highlights

- **Facebook** consistently achieves higher conversions and stronger click-conversion correlation than AdWords.
- **Statistical tests** confirm that Facebook outperforms AdWords with statistical significance.
- Regression models provide accurate predictions for conversion outcomes, supporting data-driven campaign planning.


