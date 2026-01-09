# DiMag

## Quantifying the Efficacy of Magnetic Nanoparticles for MRI and Hyperthermia Applications via Machine Learning Methods

---

[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

Built with:

![numpy](https://img.shields.io/badge/NumPy-013243.svg?style={0}&logo=NumPy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458.svg?style={0}&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikitlearn-F7931E.svg?style={0}&logo=scikit-learn&logoColor=white)
![scipy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style={0}&logo=SciPy&logoColor=white)

---

## Overview

Magnetic nanoparticles are a prospective class of materials for use in biomedicine as agents for magnetic resonance imaging (MRI) and hyperthermia treatment. However, synthesis of nanoparticles of high efficacy is resource-intensive experimental work. Machine learning accelerates the discovery and optimization of magnetic nanoparticles for biomedical applications by predicting their efficacy in MRI imaging and hyperthermia treatment.

Researchers curated a comprehensive database of over 980 magnetic nanoparticles from scientific literature and developed tree-based ensemble models achieving high predictive accuracy for specific absorption rate (SAR) and relaxivity measurements. By capturing complex relationships between nanoparticle composition, morphology, and magnetic properties, these models overcome limitations of traditional theoretical approaches. DiMag, an open-access web platform, democratizes access to trained models and the curated database, enabling rapid computational screening of nanomaterials. This work significantly reduces experimental costs and time by allowing researchers worldwide to optimize nanoparticle properties computationally before laboratory synthesis, advancing personalized medicine through improved diagnostic and therapeutic agents.

---

## Table of Contents

- [Overview](#overview)
- [Content](#content)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

---

## Content

DiMag is a machine learning framework for predicting magnetic nanoparticle efficacy in biomedical applications. The project integrates four interconnected components:

- **Curated Database**: Over 980 nanoparticles with experimentally-derived properties collected from scientific articles
- **Model Selection Module**: Comparing multiple tree-based ensemble algorithms
- **Optimized Predictive Models**: For specific absorption rate (SAR) and MRI relaxivities (r1/r2)
- **Validation Framework**: Testing generalization on unseen data

The workflow standardizes data preprocessing, feature normalization, hyperparameter optimization, and cross-validation evaluation. By achieving R² scores of 0.86 for SAR and 0.75-0.78 for relaxivities, DiMag enables rapid computational screening of nanomaterial candidates, significantly reducing experimental costs and accelerating discovery of improved diagnostic and therapeutic agents for personalized medicine.

---

## Algorithms

DiMag employs tree-based ensemble machine learning algorithms for predicting magnetic nanoparticle efficacy:

- **Light Gradient Boosting Machine (LGBMRegressor)**: Predicts specific absorption rate (SAR) for hyperthermia applications, achieving R² = 0.86
- **Extremely Randomized Trees (ExtraTreesRegressor)**: Predict r1 and r2 relaxivity values for MRI, achieving R² = 0.78 and R² = 0.75 respectively

These ensemble methods aggregate multiple decision trees to capture complex nonlinear relationships between nanoparticle descriptors and biomedical performance metrics. Data preprocessing includes stratified train-test splitting using binned discretization to ensure representative sampling, MinMax normalization of features, and log-transformation of targets. Rigorous model validation employs 10-fold cross-validation with R² and RMSE metrics. Hyperparameter optimization via Optuna framework maximizes predictive accuracy. This computational strategy enables rapid in silico screening of magnetic nanomaterials, reducing experimental costs while accelerating discovery of optimized biomedical agents.

---

## Installation

Install DiMag using one of the following methods:

**Build from source:**

1. Clone the DiMag repository:
```sh
git clone https://github.com/andreygetmanov/DiMag
```

2. Navigate to the project directory:
```sh
cd DiMag
```

3. Install the project dependencies:
```sh
pip install -r requirements.txt
```

---

## Repository Structure

The repository is organized as follows:

- `database`: CSV files of manually collected and processed databases for training and testing of ML models
- `model_selection`: Python files for ML models involved in model selection
- `models`: Python files with ML models of the best performance
- `validation`: Python files for the best ML models and CSV data files used in the validation process

---

## Citation

If you use this software, please cite it as below.

### APA format:

```
andreygetmanov (2026). DiMag repository [Computer software]. https://github.com/andreygetmanov/DiMag
```

### BibTeX format:

```bibtex
@misc{DiMag,
    author = {andreygetmanov},
    title = {DiMag repository},
    year = {2026},
    publisher = {github.com},
    journal = {github.com repository},
    howpublished = {\url{https://github.com/andreygetmanov/DiMag.git}},
    url = {https://github.com/andreygetmanov/DiMag.git}
}
```