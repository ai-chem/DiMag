# Quantifying the efficiency of magnetic nanoparticles for MRI and hyperthermia applications via machine learning methods

Created Machine Learning models quantitative predict SAR (W/g) value (LBMBRegressor with Q<sup>2</sup>=0.86 on 10-fold cross-validation), and r<sub>1</sub>/r<sub>2</sub> relaxivities (mM<sup>-1</sup>s<sup>-1</sup>) values (ExtraTreesRegressor with Q<sup>2</sup>=0.72 and Q<sup>2</sup>=0.71 respectively on 10-fold cross-validation). 

Obtained models were implemented into DiMag open-access web-resource with the expandable database with links to the sources, its visualisation tool and three levels of predictions (base, progressime and advanced). This resource will simplify the process of obtaining magnetic nanoparticles with desired efficiency in MRI and hyperthermia treatment.

![Site](https://user-images.githubusercontent.com/110278259/225967053-3fb95f6a-61ac-44db-be0a-a64e4611427f.png)

In this repository there are collected databases, created machine learning algorithms and their validation. The stricture of this repository is follows:
>Directory Database contains .csv files of manually collected and processed databases for training and testing of models.
>
>Directory Choosing_of_ML_models contains .py files for models seen during the selection process.
>
>Directory ML_models contains .py files for chosen ML models with the best performance.
>
>Directory Validation contains .py files for chosen ML models and .csv files of data used in the validation process.
