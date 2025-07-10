# DiMag

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

DiMag is designed to advance the development of magnetic nanoparticles used in medical imaging and treatment by predicting important properties that determine their effectiveness. By gathering extensive data from numerous scientific studies, it combines detailed information about nanoparticle composition, surface characteristics, and shape to build accurate prediction models. These models help estimate how nanoparticles will behave in treatments like MRI scans and hyperthermia therapy. Users can benefit from reliable predictions and clear visual comparisons between expected and actual results. The system also offers easy access to a large, curated database and supports the addition of new data, promoting collaboration among researchers worldwide. By simplifying complex data analysis and making it widely accessible, DiMag aims to speed up research, improve experimental planning, and foster innovative approaches in designing magnetic nanomaterials for healthcare applications.

Magnetic nanoparticles are a prospective class of materials for use in biomedicine as agents for magnetic resonance imaging (MRI) and hyperthermia treatment. However, synthesis of nanoparticles of high efficacy is resource-intensive experimental work. In turn, the use of machine learning (ML) methods is becoming useful in materials design and serves as a great approach to designing nanomagnets for biomedicine. In this work, for the first time we develop an ML-based approach for prediction of main parameters of material efficacy, i.e., specific absorption rate (SAR) for hyperthermia and r1/r2 relaxivities in MRI, with parameters of nanoparticles as well as experimental conditions as descriptors. For that, we assemble a unique database with more than 980 magnetic nanoparticles collected from scientific articles. Using this data, we train several tree-based ensemble models to predict SAR, r1 and r2 relaxivity. After hyperparameter optimization, models reach performance of R² = 0.86, R² = 0.78 and R² = 0.75, respectively. Testing the models on samples unseen during the training shows no performance drops. Finally, we develop [DiMag](http://dimag.acidlab.space), an open-access resource created to guide synthesis of novel nanosized magnets for MRI and hyperthermia treatment with machine learning and boost development of new biomedical agents.

---

## Repository content

The DiMag repository is structured to support advanced machine learning-based prediction and analysis of magnetic nanoparticle properties relevant for biomedical applications such as MRI diagnosis and hyperthermia treatment. The core components include a comprehensive database and multiple predictive modeling scripts.

- `database`: CSV files of manually collected and processed databases for training and testing of ML models.
- `model_selection`: Python files for ML models involved in model selection.
- `models`: Python files with ML models of the best performance.
- `validation`: Python files for the best ML models and CSV data files used in the validation process.

The database component stores curated experimental data on magnetic nanoparticles, featuring key metrics like specific absorption rate (SAR) and relaxivities (r1 and r2), along with engineered descriptors reflecting their composition, surface chemistry, and morphology. These datasets form the foundation for model training and validation. The modeling components comprise scripts implementing various machine learning regressors—including ExtraTreesRegressor, RandomForestRegressor, LightGBM, and XGBoost—each dedicated to predicting specific nanoparticle properties based on the curated data.

These models are trained and validated through stratified data splitting, normalization, and rigorous cross-validation methods to ensure robust predictive performance. Visualization routines embedded in the scripts facilitate the assessment of model accuracy and reliability. Together, the database and modeling modules integrate seamlessly to enable the generation of accurate predictive models that capture the complex relationships governing nanoparticle behavior.

Additionally, the repository underpins a publicly available web service, DiMag, that allows researchers worldwide to access the database and predictive tools, submit new data, and perform in silico screenings. This interconnected design accelerates data-driven optimization of magnetic nanomaterials, fosters open collaborative research, and supports the translation of experimental insights into practical biomedical applications.

---

## Used algorithms

The codebase employs several ensemble regression algorithms to predict the specific absorption rate (SAR) and relaxivity properties (r1 and r2) of magnetic nanoparticles based on their molecular and particle descriptors. These algorithms include Extra Trees, Random Forest, LightGBM, and XGBoost. Each of these algorithms creates multiple decision trees and combines their outputs to improve the accuracy and robustness of predictions.

They help capture complex relationships between input features and the target properties. The models are optimized with hyperparameter tuning to enhance performance. Additionally, data preprocessing steps such as stratified data splitting, feature normalization, and transformation of target values are used to prepare the data effectively before applying these algorithms. Cross-validation techniques assess the models' reliability, while evaluation metrics like R², MAE, MSE, and RMSE measure predictive accuracy.

Overall, these algorithms contribute to building reliable and interpretable models that support the prediction of important bio-physical properties relevant to biomedical applications.

---