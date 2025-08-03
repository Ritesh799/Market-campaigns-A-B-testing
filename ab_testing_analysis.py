# 1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# For cointegration test (if desired)
from statsmodels.tsa.stattools import coint

# 2. Load the dataset
df = pd.read_csv('marketing_campaign.csv')

# 3. Rename columns to consistent snake_case
rename_dict = {
    'Date': 'date',
    'Facebook Ad Campaign': 'facebook_ad_campaign',
    'Facebook Ad Views': 'facebook_ad_views',
    'Facebook Ad Clicks': 'facebook_ad_clicks',
    'Facebook Ad Conversions': 'facebook_ad_conversions',
    'Cost per Facebook Ad': 'facebook_cost_per_ad',
    'Facebook Click-Through Rate (Clicks / View)': 'facebook_ctr',
    'Facebook Conversion Rate (Conversions / Clicks)': 'facebook_conversion_rate',
    'Facebook Cost per Click (Ad Cost / Clicks)': 'facebook_cost_per_click',
    'AdWords Ad Campaign': 'adword_ad_campaign',
    'AdWords Ad Views': 'adword_ad_views',
    'AdWords Ad Clicks': 'adword_ad_clicks',
    'AdWords Ad Conversions': 'adword_ad_conversions',
    'Cost per AdWords Ad': 'adword_cost_per_ad',
    'AdWords Click-Through Rate (Clicks / View)': 'adword_ctr',
    'AdWords Conversion Rate (Conversions / Click)': 'adword_conversion_rate',
    'AdWords Cost per Click (Ad Cost / Clicks)': 'adword_cost_per_click'
}
df.rename(columns=rename_dict, inplace=True)

# 4. Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# 5. Define cleaning functions that handle mixed types gracefully

def clean_percentage(x):
    if pd.isnull(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x.endswith('%'):
            x = x[:-1]
        try:
            return float(x) / 100
        except:
            return np.nan
    return float(x)

def clean_currency(x):
    if pd.isnull(x):
        return np.nan
    if isinstance(x, str):
        x = x.strip()
        if x.startswith('$'):
            x = x[1:]
        x = x.replace(',', '')  # Remove commas in numbers like "1,234.56"
        try:
            return float(x)
        except:
            return np.nan
    return float(x)

# 6. Clean relevant columns
df['facebook_ctr'] = df['facebook_ctr'].apply(clean_percentage)
df['facebook_conversion_rate'] = df['facebook_conversion_rate'].apply(clean_percentage)
df['facebook_cost_per_click'] = df['facebook_cost_per_click'].apply(clean_currency)
df['facebook_cost_per_ad'] = df['facebook_cost_per_ad'].apply(clean_currency)

df['adword_ctr'] = df['adword_ctr'].apply(clean_percentage)
df['adword_conversion_rate'] = df['adword_conversion_rate'].apply(clean_percentage)
df['adword_cost_per_click'] = df['adword_cost_per_click'].apply(clean_currency)
df['adword_cost_per_ad'] = df['adword_cost_per_ad'].apply(clean_currency)

# 7. Exploratory Data Analysis: distribution histograms for clicks and conversions

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.title('Facebook Ad Clicks')
sns.histplot(df['facebook_ad_clicks'], bins=20, kde=True)
plt.subplot(1, 2, 2)
plt.title('Facebook Ad Conversions')
sns.histplot(df['facebook_ad_conversions'], bins=20, kde=True)
plt.show()

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.title('AdWords Ad Clicks')
sns.histplot(df['adword_ad_clicks'], bins=20, kde=True)
plt.subplot(1, 2, 2)
plt.title('AdWords Ad Conversions')
sns.histplot(df['adword_ad_conversions'], bins=20, kde=True)
plt.show()

# 8. Define function to bucket conversions into categories

def create_conversion_category(series):
    category = []
    for conv in series:
        if conv < 6:
            category.append('less than 6')
        elif 6 <= conv < 11:
            category.append('6 - 10')
        elif 11 <= conv < 16:
            category.append('10 - 15')
        else:
            category.append('more than 15')
    return category

df['facebook_conversion_category'] = create_conversion_category(df['facebook_ad_conversions'])
df['adword_conversion_category'] = create_conversion_category(df['adword_ad_conversions'])

# 9. Count number of days per conversion category for Facebook and AdWords

fb_count = df['facebook_conversion_category'].value_counts().reset_index()
fb_count.columns = ['Category', 'Facebook']

adw_count = df['adword_conversion_category'].value_counts().reset_index()
adw_count.columns = ['Category', 'AdWords']

category_df = pd.merge(fb_count, adw_count, on='Category', how='outer').fillna(0)

# Order the categories
categories_order = ['less than 6', '6 - 10', '10 - 15', 'more than 15']
category_df['Category'] = pd.Categorical(category_df['Category'], categories=categories_order, ordered=True)
category_df = category_df.sort_values('Category').reset_index(drop=True)
print(category_df)

# 10. Plot conversion categories comparison

x_axis = np.arange(len(category_df))
plt.figure(figsize=(12,6))
plt.bar(x_axis - 0.2, category_df['Facebook'], 0.4, label='Facebook', color='#03989E', edgecolor='k')
plt.bar(x_axis + 0.2, category_df['AdWords'], 0.4, label='AdWords', color='#A62372', edgecolor='k')
plt.xticks(x_axis, category_df['Category'])
plt.xlabel('Conversion Category')
plt.ylabel('Number of Days')
plt.title('Daily Conversion Frequency by Conversion Category')
plt.legend()
plt.show()

# 11. Scatterplots Clicks vs Conversions

plt.figure(figsize=(15, 6))
plt.subplot(1,2,1)
plt.title('Facebook: Clicks vs Conversions')
sns.scatterplot(x=df['facebook_ad_clicks'], y=df['facebook_ad_conversions'], color='#03989E')
plt.xlabel('Clicks')
plt.ylabel('Conversions')

plt.subplot(1,2,2)
plt.title('AdWords: Clicks vs Conversions')
sns.scatterplot(x=df['adword_ad_clicks'], y=df['adword_ad_conversions'], color='#A62372')
plt.xlabel('Clicks')
plt.ylabel('Conversions')
plt.show()

# 12. Correlation coefficients

fb_corr = df[['facebook_ad_clicks', 'facebook_ad_conversions']].corr().iloc[0,1]
adw_corr = df[['adword_ad_clicks', 'adword_ad_conversions']].corr().iloc[0,1]
print(f"Facebook correlation (clicks vs conversions): {fb_corr:.2f}")
print(f"AdWords correlation (clicks vs conversions): {adw_corr:.2f}")

# 13. Hypothesis Testing (Welch's t-test)

print('Mean Conversions:')
print(f"Facebook: {df['facebook_ad_conversions'].mean():.2f}")
print(f"AdWords: {df['adword_ad_conversions'].mean():.2f}")

t_stat, p_val = st.ttest_ind(df['facebook_ad_conversions'], df['adword_ad_conversions'], equal_var=False)
print(f"t-statistic: {t_stat:.2f}")
print(f"p-value: {p_val:.4e}")

if p_val < 0.05:
    print("Reject null hypothesis: Facebook conversions are significantly higher than AdWords.")
else:
    print("Do not reject null hypothesis: No significant difference in conversions.")

# 14. Linear Regression for Facebook: predict conversions from clicks

X = df[['facebook_ad_clicks']]
y = df[['facebook_ad_conversions']]
reg_model = LinearRegression()
reg_model.fit(X, y)
y_pred = reg_model.predict(X)

r2 = r2_score(y, y_pred)*100
mse = mean_squared_error(y, y_pred)

print(f"Linear Regression Results (Facebook)")
print(f"R^2 Score: {r2:.2f}%")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Intercept: {reg_model.intercept_[0]:.2f}")
print(f"Coefficient: {reg_model.coef_[0][0]:.2f}")

plt.figure(figsize=(8,6))
sns.scatterplot(x=df['facebook_ad_clicks'], y=df['facebook_ad_conversions'], color='#03989E', label='Actual Data')
plt.plot(df['facebook_ad_clicks'], y_pred, color='#A62372', label='Regression Line')
plt.xlabel('Facebook Ad Clicks')
plt.ylabel('Facebook Ad Conversions')
plt.title('Linear Regression: Facebook Clicks vs Conversions')
plt.legend()
plt.show()

# Predict expected conversions for given clicks
clicks_to_predict = [50, 80]
for clicks in clicks_to_predict:
    pred = reg_model.predict([[clicks]])[0][0]
    print(f"Expected Facebook conversions for {clicks} clicks: {pred:.2f}")

# 15. Time-based analysis: Add month and week day columns

df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday  # Monday=0

# Plot weekly total Facebook conversions
weekly_conv = df.groupby('weekday')['facebook_ad_conversions'].sum()
weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

plt.figure(figsize=(8,5))
plt.bar(weekdays, weekly_conv, color='#03989E', edgecolor='k')
plt.title('Weekly Facebook Conversions')
plt.xlabel('Weekday')
plt.ylabel('Total Conversions')
plt.show()

# Plot monthly total Facebook conversions
monthly_conv = df.groupby('month')['facebook_ad_conversions'].sum()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(8,5))
plt.plot(month_names, monthly_conv, '-o', color='#A62372')
plt.title('Monthly Facebook Conversions')
plt.xlabel('Month')
plt.ylabel('Total Conversions')
plt.show()

# 16. Calculate Cost per Conversion (CPC)

df['facebook_cpc'] = df['facebook_cost_per_ad'] / df['facebook_ad_conversions'].replace(0, np.nan)
df['adword_cpc'] = df['adword_cost_per_ad'] / df['adword_ad_conversions'].replace(0, np.nan)

print("Facebook CPC Summary:")
print(df['facebook_cpc'].describe())

print("AdWords CPC Summary:")
print(df['adword_cpc'].describe())

# 17. (Optional) Cointegration test between Cost and Conversions (Facebook example)

score, pvalue, _ = coint(df['facebook_cost_per_ad'], df['facebook_ad_conversions'])
print(f"Cointegration test score: {score}")
print(f"P-value: {pvalue}")

if pvalue < 0.05:
    print("Reject null hypothesis: advertising costs and conversions are cointegrated (long-term equilibrium).")
else:
    print("Do not reject null hypothesis: no cointegration found.")

# End of analysis script
