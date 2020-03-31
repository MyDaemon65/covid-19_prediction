#########################################################################

# COVID-19 Machine Learning Prediction Program
# Matt Taylor
# 03/29/2020
# MJT65@ProtonMail.com

########################################################################


#Import Required Modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import CSV file from John Hopkins
COVID_CONFIRMED_URL  = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
covid_confirmed = pd.read_csv(COVID_CONFIRMED_URL)

# Print HEAD of CSV Import File
# print(covid_confirmed.head())

covid_confirmed_long = pd.melt(covid_confirmed,
                               id_vars=covid_confirmed.iloc[:, :4],
                               var_name='date',
                               value_name='confirmed')

covid_confirmed_long['Country/Region'].replace('Mainland China', 'China', inplace=True)
covid_confirmed_long[['Province/State']] = covid_confirmed_long[['Province/State']].fillna('')
covid_confirmed_long.fillna(0, inplace=True)
covid_confirmed_long.isna().sum().sum()
covid_countries_date_df = covid_confirmed_long.groupby(['Country/Region', 'date'], sort=False).sum().reset_index()
covid_countries_date_df.drop(['Lat', 'Long'], axis=1, inplace=True)

# Filter out by Country
COUNTRY = 'US'
covid_country = covid_countries_date_df[covid_countries_date_df['Country/Region'] == COUNTRY]
days = np.array([i for i in range(len(covid_country['date']))])

# Skip Days the First 30 Days After January 22/2020
SKIP_DAYS = 30
covid_country_confirmed_sm = list(covid_country['confirmed'][SKIP_DAYS:])
covid_country_confirmed_sm[:15]
X = days[SKIP_DAYS:].reshape(-1, 1)
y = list(np.log(covid_country_confirmed_sm))

#Train The Data
X_train, X_test, y_train, y_test = train_test_split(X, y,
         test_size=0.1,
         shuffle=False)

linear_model = LinearRegression(fit_intercept=True)
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

# Print Mean Absolute & Mean Standard Error Rate
print('MAE:', mean_absolute_error(y_pred, y_test))
print('MSE:',mean_squared_error(y_pred, y_test))

a = linear_model.coef_
b = linear_model.intercept_
X_fore = list(np.arange(len(days), len(days) + 14))
y_fore = [(a*x+b)[0] for x in X_fore]

y_train_l = list(np.exp(y_train))
y_test_l = list(np.exp(y_test))
y_pred_l = list(np.exp(y_pred))
y_fore_l = list(np.exp(y_fore))

#Linear Scale with Predictions
fig, ax = plt.subplots(figsize=(16, 6))

sns.lineplot(x=days, y=covid_country['confirmed'],
             markeredgecolor="#2980b9", markerfacecolor="#2980b9", markersize=8, marker="o",
             sort=False, linewidth=1, color="#2980b9")

sns.lineplot(x=X_train.reshape(-1), y=y_train_l,
             markeredgecolor="#051118", markerfacecolor="#051118", markersize=8, marker="o",
             sort=False, linewidth=1, color="#3498db")

sns.lineplot(x=X_test.reshape(-1), y=y_test_l,
             markeredgecolor="#e67e22", markerfacecolor="#e67e22", markersize=8, marker="o",
             sort=False, linewidth=1, color="#e67e22")

sns.lineplot(x=X_test.reshape(-1), y=y_pred_l,
             markeredgecolor="#f1c40f", markerfacecolor="#f1c40f", markersize=8, marker="o",
             sort=False, linewidth=1, color="#f1c40f")

sns.lineplot(x=X_fore, y=y_fore_l,
             markeredgecolor="#2ecc71", markerfacecolor="#2ecc71", markersize=8, marker="o",
             sort=False, linewidth=1, color="#2ecc71")

plt.suptitle(f"COVID-19 confirmed cases and forecasting in {COUNTRY} over the time", fontsize=16, fontweight='bold', color='white')

plt.ylabel('Confirmed cases')
plt.xlabel('Days since 1/22')

plt.legend(['Unused train data', 'Train data', 'Test data', 'Predictions', 'Forecast'])
plt.savefig('reg.svg', format='svg', dpi=1200)
plt.show()

