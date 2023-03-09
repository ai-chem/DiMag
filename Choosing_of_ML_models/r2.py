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
df = pd.read_csv('../Database/r2.csv')
df = df.loc[:, 'Area/volume':'r2 relaxivity']



#splitting data into descriptors (features) and predicted (r2 relaxivity) value
features = df.copy()
features = df.drop('r2 relaxivity', axis=1).columns.to_list()
x = df.loc[:, features].values
y = df.loc[:, ['r2 relaxivity']].values

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
          RandomForestRegressor(min_samples_leaf=1, n_estimators=124),
		  LGBMRegressor(),
		  XGBRegressor(n_estimators=98, max_depth=2,
                       learning_rate=0.39162340799554574, min_child_weight=0.0852067889182647,
                       reg_alpha=0.6866245565279169, reg_lambda=0.5217971351407126)]:
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
    plt.xlim(0.8, 3.2)
    plt.ylim(0.8, 3.2)
    plt.title(models[k-1], loc='left', fontsize=15, fontweight='bold')
    plt.text(2, 1.75, f'cross-validation (r2): {round(accuracy.mean(), 2)}', fontsize=10)
    plt.text(2, 1.65, f'cross-validation (RMSE): {round(accuracy1.mean(), 2)}', fontsize=10)
    plt.text(2, 1.55, f'r2_test: {round(metrics.r2_score(y_test, y_pred), 2)}', fontsize=10)
    plt.text(2, 1.45, f'MAE_test: {round(metrics.mean_absolute_error(y_test, y_pred), 2)}', fontsize=10)
    plt.text(2, 1.35, f'MSE_test: {round(metrics.mean_squared_error(y_test, y_pred), 2)}', fontsize=10)
    plt.text(2, 1.25, f'RMSE_test: {round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2)}', fontsize=10)
    plt.text(2, 1.15, f'r2_train: {round(metrics.r2_score(y_train, y1_pred), 2)}', fontsize=10)
    plt.text(2, 1.05, f'MAE_train: {round(metrics.mean_absolute_error(y_train, y1_pred), 2)}', fontsize=10)
    plt.text(2, 0.95, f'MSE_train: {round(metrics.mean_squared_error(y_train, y1_pred), 2)}', fontsize=10)
    plt.text(2, 0.85, f'RMSE_train: {round(np.sqrt(metrics.mean_squared_error(y_train, y1_pred)), 2)}', fontsize=10)
    k += 1
plt.show()