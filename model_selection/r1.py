from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor



matplotlib.use('TkAgg')

#importing data from .csv file
df = pd.read_csv('../database/r1.csv')
df = df.loc[:, 'Area/volume':'r1 relaxivity']

#splitting data into descriptors (features) and the predicted (r1 relaxivity) values
features = df.copy()
features = df.drop('r1 relaxivity', axis=1).columns.to_list()
x = df.loc[:, features].values
y = df.loc[:, ['r1 relaxivity']].values

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

#plotting of ML_models
plt.figure(figsize=[40, 30])
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

models = ['ExtraTreesRegressor', 'RandomForestRegressor', 'LGBMRegressor', 'XGBRegressor']
k = 1
for i in [ExtraTreesRegressor(min_samples_leaf=2, n_estimators=102),
          RandomForestRegressor(min_samples_leaf=2, n_estimators=102),
          LGBMRegressor(n_estimators=98, max_depth=2, learning_rate=0.5370528270452327, min_split_gain=0.00012531418525182567, min_child_weight=0.07922350135025907, min_child_samples=11, reg_alpha=0.30270847225705616, reg_lambda=0.05726625934217654),
          XGBRegressor(n_estimators=78, max_depth=2, learning_rate=0.5830574529470156, min_child_weight=0.07576116552722682, reg_alpha=0.3111727161212538, reg_lambda=0.501606756809154)]:
    # regression model
    regressor = i.fit(x_train, y_train)
    regressor.score(x_test, y_test)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    y1_pred = regressor.predict(x_train)
    # cross validation
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(regressor, x_train, y_train, cv=10,
                               scoring='r2')
    accuracy1 = cross_val_score(regressor, x_train, y_train, cv=10,
                                scoring='neg_root_mean_squared_error')
    print("10-fold cross-validation (r2):", accuracy.mean())
    print("10-fold cross-validation (RMSE):", accuracy1.mean())
    # accuracy of model
    from sklearn import metrics
    print('r2_test:', metrics.r2_score(y_test, y_pred))
    print('MAE_test:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE_test:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE_test:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('r2_train:', metrics.r2_score(y_train, y1_pred))
    print('MAE_train:', metrics.mean_absolute_error(y_train, y1_pred))
    print('MSE_train:', metrics.mean_squared_error(y_train, y1_pred))
    print('RMSE_train:', np.sqrt(metrics.mean_squared_error(y_train, y1_pred)))
    ## Plot the subplots
    plt.subplot(2, 2, k)
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.spines.top'] = False
    plt.scatter(y_train, y1_pred, color='#3a0170', s=10)
    plt.scatter(y_test, y_pred, color='#b15fff', s=10)
    plt.plot(y_test, y_test, color='black')
    plt.xlabel('test data', size=15)
    plt.ylabel('predicted data', size=15)
    plt.tick_params(axis='both', labelsize=10)
    plt.xlim(-1, 3)
    plt.ylim(-1, 2)
    plt.title(models[k-1], loc='left', fontsize=15, fontweight='bold')
    plt.text(1, 0.5, f'cross-validation (r2): {round(accuracy.mean(), 2)}', fontsize=10)
    plt.text(1, 0.35, f'cross-validation (RMSE): {round(accuracy1.mean(), 2)}', fontsize=10)
    plt.text(1, 0.20, f'r2_test: {round(metrics.r2_score(y_test, y_pred), 2)}', fontsize=10)
    plt.text(1, 0.05, f'MAE_test: {round(metrics.mean_absolute_error(y_test, y_pred), 2)}', fontsize=10)
    plt.text(1, -0.10, f'MSE_test: {round(metrics.mean_squared_error(y_test, y_pred), 2)}', fontsize=10)
    plt.text(1, -0.25, f'RMSE_test: {round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2)}', fontsize=10)
    plt.text(1, -0.40, f'r2_train: {round(metrics.r2_score(y_train, y1_pred), 2)}', fontsize=10)
    plt.text(1, -0.55, f'MAE_train: {round(metrics.mean_absolute_error(y_train, y1_pred), 2)}', fontsize=10)
    plt.text(1, -0.70, f'MSE_train: {round(metrics.mean_squared_error(y_train, y1_pred), 2)}', fontsize=10)
    plt.text(1, -0.85, f'RMSE_train: {round(np.sqrt(metrics.mean_squared_error(y_train, y1_pred)), 2)}', fontsize=10)
    k += 1
plt.show()
