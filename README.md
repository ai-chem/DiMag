# DiMag

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

DiMag is designed to advance the design and optimization of magnetic nanoparticles used in medical imaging and treatment, specifically for magnetic resonance imaging (MRI) and hyperthermia applications. Magnetic nanoparticles are promising materials in biomedicine, but synthesizing highly effective nanoparticles requires resource-intensive experimental work.

DiMag provides a sophisticated approach that leverages extensive experimental data collected from over 980 magnetic nanoparticles sourced from scientific articles. Through careful processing of this data—including selecting key parameters of nanoparticles and experimental conditions—DiMag employs advanced machine learning techniques to create accurate predictive models for crucial material efficacy parameters: specific absorption rate (SAR) for hyperthermia and r1/r2 relaxivities for MRI.

Using ensemble tree-based models such as ExtraTreesRegressor, RandomForestRegressor, LightGBM, and XGBoost, the models achieve strong predictive performance with R² values of 0.86 for SAR, 0.78 for r1 relaxivity, and 0.75 for r2 relaxivity after hyperparameter optimization. Testing on unseen samples confirms model robustness without performance degradation.

The platform offers tools for visualization and comparison of model predictions, facilitating interpretation of results and identification of promising nanoparticle candidates. Additionally, DiMag includes a publicly available web service ([DiMag](http://dimag.acidlab.space)) that enables rapid screening and exploration of magnetic nanoparticle data and models, promoting community collaboration and ongoing improvement.

By enabling reliable predictions and insights, DiMag aims to accelerate the development of effective magnetic nanomaterials for biomedical applications, complementing experimental research with efficient computational methods.

## Repository content

The DiMag repository is structured to facilitate the design and optimization of magnetic nanoparticles for biomedical applications through machine learning. It contains the following components:

- `database`: CSV files of manually collected and processed databases for training and testing of ML models. This unique dataset incorporates physicochemical properties of nanoparticles and experimental conditions.
- `model_selection`: Python scripts implementing multiple machine learning models involved in the model selection process.
- `models`: Python scripts containing the best-performing machine learning models used for final predictions.
- `validation`: Python scripts and CSV data files used for validating the best ML models.

The repository integrates dataset curation, feature engineering, model training, hyperparameter optimization, and performance evaluation—all aiming to provide robust, data-driven predictions of nanoparticle behavior. Visualization scripts within this framework support graphical analysis to interpret model outcomes and assess prediction accuracy.

## Used algorithms

DiMag employs several ensemble regression algorithms to predict key properties of magnetic nanoparticles:

- **ExtraTreesRegressor** and **RandomForestRegressor**: These build multiple decision trees and aggregate their outputs to improve prediction accuracy and stability.
- **LightGBM** and **XGBoost**: These are advanced gradient boosting algorithms that sequentially build models where each new model corrects the errors of the previous one, enhancing overall performance.

These algorithms analyze complex experimental data to predict metrics such as the specific absorption rate (SAR) and r1/r2 relaxivities. Together, they enable understanding and optimization of nanoparticle behavior through machine learning.