from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd




matplotlib.use('TkAgg')

#importing data from .csv file
df = pd.read_csv('../Database/SAR.csv')
df = df.loc[:, 'Concentration of particles':'SAR']

#splitting data into descriptors (features) and predicted (SAR) values
features = df.copy()
features = df.drop('SAR', axis=1).columns.to_list()
x = df.loc[:, features].values
y = df.loc[:, ['SAR']].values

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

#Machine Learning algorithm
from lightgbm import LGBMRegressor
regressor = LGBMRegressor(n_estimators=100, max_depth=2,
                          learning_rate=0.26485162360302365, min_split_gain=9.598896733776042e-05,
                          min_child_weight=0.031405041091885896, min_child_samples=14,
                          reg_alpha=0.15151696313077256, reg_lambda=0.13651480472249508)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
y1_pred = regressor.predict(x_train)

#cross validation procedure
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(regressor, x_train, y_train, cv=10, scoring='r2')
accuracy1 = abs(cross_val_score(regressor, x_train, y_train, cv=10, scoring='neg_root_mean_squared_error'))
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
plt.scatter(y_test, y_pred, color='#b15fff', s=70)
plt.plot(y_test, y_test, color='gray')
ci = abs(accuracy1.mean())
x = np.linspace(-1, 4, 110)
ax.fill_between(x, (x-ci), (x+ci), color='b', alpha=.3)
plt.title('LGBMRegressor SAR', size=30)
plt.xlabel('test data', fontsize=20, family='Arial')
plt.ylabel('predicted data', fontsize=20, family='Arial')
plt.tick_params(axis='both', labelsize=20)
plt.xlim(-1, 4)
plt.ylim(-1, 4)
plt.text(1.8, 0.9, f'R2 cross-validation: {round(accuracy.mean(), 2)}', fontsize=25, family='Arial')
plt.text(1.8, 0.65 , f'RMSE cross-validation: {round(abs(accuracy1.mean()), 2)}', fontsize=25, family='Arial')
plt.text(1.8, 0.4, f'R2 test: {round(metrics.r2_score(y_test, y_pred), 2)}', fontsize=25, family='Arial')
plt.text(1.8, 0.15, f'RMSE test: {round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2)}', fontsize=25, family='Arial')
plt.text(1.8, -0.1, f'R2 train: {round(metrics.r2_score(y_train, y1_pred), 2)}', fontsize=25, family='Arial')
plt.text(1.8, -0.35, f'RMSE train: {round(np.sqrt(metrics.mean_squared_error(y_train, y1_pred)), 2)}', fontsize=25, family='Arial')
plt.show()
