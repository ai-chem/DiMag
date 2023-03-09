from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


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

#plotting of ML_models
plt.figure(figsize=[40, 30])
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

models = ['ExtraTreesRegressor', 'RandomForestRegressor', 'LGBMRegressor', 'XGBRegressor']
k = 1
for i in [ExtraTreesRegressor(min_samples_leaf=2, n_estimators=105),
          RandomForestRegressor(min_samples_leaf=2, n_estimators=127),
          LGBMRegressor(n_estimators=100, max_depth=2, learning_rate=0.26485162360302365, min_split_gain=9.598896733776042e-05, min_child_weight=0.031405041091885896, min_child_samples=14, reg_alpha=0.15151696313077256, reg_lambda=0.13651480472249508),
		  XGBRegressor(n_estimators=119, max_depth=2, learning_rate=0.4893964080964038, min_child_weight=0.030887991255315807, reg_alpha=0.03154398899882912, reg_lambda=0.14994515092556174)]:
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

    # accuracy of the model
    from sklearn import metrics
    print('r2_test:', metrics.r2_score(y_test, y_pred))
    print('MAE_test:', metrics.mean_absolute_error(y_test, y_pred))
    print('MSE_test:', metrics.mean_squared_error(y_test, y_pred))
    print('RMSE_test:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('r2_train:', metrics.r2_score(y_train, y1_pred))
    print('MAE_train:', metrics.mean_absolute_error(y_train, y1_pred))
    print('MSE_train:', metrics.mean_squared_error(y_train, y1_pred))
    print('RMSE_train:', np.sqrt(metrics.mean_squared_error(y_train, y1_pred)))

    # plot the subplots
    plt.subplot(2, 2, k)
    plt.scatter(y_train, y1_pred, color='#3a0170', s=10)
    plt.scatter(y_test, y_pred, color='#b15fff', s=10)
    plt.plot(y_test, y_test, color='black')
    plt.xlabel('test data', size=15)
    plt.ylabel('predicted data', size=15)
    plt.tick_params(axis='both', labelsize=10)
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.title(models[k-1], loc='left', fontsize=15, fontweight='bold')
    plt.text(2, 1.3, f'cross-validation (r2): {round(accuracy.mean(), 2)}', fontsize=10)
    plt.text(2, 1.1, f'cross-validation (RMSE): {round(accuracy1.mean(), 2)}', fontsize=10)
    plt.text(2, 0.9, f'r2_test: {round(metrics.r2_score(y_test, y_pred), 2)}', fontsize=10)
    plt.text(2, 0.7, f'MAE_test: {round(metrics.mean_absolute_error(y_test, y_pred), 2)}', fontsize=10)
    plt.text(2, 0.5, f'MSE_test: {round(metrics.mean_squared_error(y_test, y_pred), 2)}', fontsize=10)
    plt.text(2, 0.3, f'RMSE_test: {round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2)}', fontsize=10)
    plt.text(2, 0.1, f'r2_train: {round(metrics.r2_score(y_train, y1_pred), 2)}', fontsize=10)
    plt.text(2, -0.1, f'MAE_train: {round(metrics.mean_absolute_error(y_train, y1_pred), 2)}', fontsize=10)
    plt.text(2, -0.3, f'MSE_train: {round(metrics.mean_squared_error(y_train, y1_pred), 2)}', fontsize=10)
    plt.text(2, -0.5, f'RMSE_train: {round(np.sqrt(metrics.mean_squared_error(y_train, y1_pred)), 2)}', fontsize=10)
    k += 1
plt.show()