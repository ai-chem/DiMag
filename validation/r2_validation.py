from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

matplotlib.use('TkAgg')

#importing data from .csv file
df = pd.read_csv('../database/r2.csv')
df = df.loc[:, 'Area/volume':'r2 relaxivity']
df = df.rename(columns={'Magnetic moment of coating': 'Magnetic moment of shell'})

df_valid = pd.read_csv('r1r2_validation.csv')

#splitting data into predictors and predicted
features = df.copy()
features = df.drop('r2 relaxivity', axis=1).columns.to_list()
x = df.loc[:, features].values
y = df.loc[:, ['r2 relaxivity']].values

x_valid = df_valid[features].values
y_valid = df_valid['Log10(r2)']

#splitting data using stratification procedure
y_discretized = KBinsDiscretizer(n_bins=5,
                                 encode='ordinal',
                                 strategy='uniform').fit_transform(y.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(),
                                                  test_size=0.2,
                                                  random_state=10,
                                                 stratify=y_discretized)

#normalizing descriptors and taking Log10 from the predicted value
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
y_train = np.log10(y_train)
y_test = np.log10(y_test)
x_valid = sc.transform(x_valid)

#Machine Learning algorithm
from sklearn.ensemble import ExtraTreesRegressor
regressor = ExtraTreesRegressor(min_samples_leaf=2, n_estimators=102)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
y1_pred = regressor.predict(x_train)
y_valid_predict = regressor.predict(x_valid)

#cross validation procedure
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(regressor, x_train, y_train, cv=10, scoring='r2')
accuracy1 = cross_val_score(regressor, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error')
print("10-fold cross-validation (r2):", accuracy.mean())
print("10-fold cross-validation (RMSE):", abs(accuracy1.mean()))

#accuracy of obtained model
from sklearn import metrics
print('r2_test:', metrics.r2_score(y_test, y_pred))
print('MAE_test:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE_test:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE_test:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r2_train:', metrics.r2_score(y_train, y1_pred))
print('MAE_train:', metrics.mean_absolute_error(y_train, y1_pred))
print('MSE_train:', metrics.mean_squared_error(y_train, y1_pred))
print('RMSE_train:', np.sqrt(metrics.mean_squared_error(y_train, y1_pred)))

#plotting of results
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
f, ax = plt.subplots(figsize=(13, 10))
plt.scatter(y_train, y1_pred, color='#3a0170', s=70)
plt.scatter(y_test, y_pred, color='#b15fff',s=70)
plt.scatter(y_valid, y_valid_predict, color='#FF0000', s=70)
plt.plot(y_test, y_test, color='gray')
ci = abs(accuracy1.mean())
x = np.linspace(-4, 4, 110)
ax.fill_between(x, (x-ci), (x+ci), color='b', alpha=.3)
plt.title('ExtraTreesRegressor r2 relaxivity', size=30)
plt.xlabel('test data', fontsize=20, family='Arial')
plt.ylabel('predicted data', fontsize=20, family='Arial')
plt.tick_params(axis='both', labelsize=20)
plt.xlim(0.8, 3.2)
plt.ylim(0.8, 3.2)
plt.text(1.6, 1.2, f'RMSE cross-validation: {round(abs(accuracy1.mean()), 2)}', fontsize=25, family='Arial')
plt.text(1.6, 0.9, f'Mean difference between predicted and \ntrue values for validation samples: {round(abs(y_valid - y_valid_predict).mean(), 2)}', fontsize=25, family='Arial')
plt.show()
