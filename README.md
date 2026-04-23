# ASTRAI
* Source code for project ASTRAI

Repository for the ASTRAI project, dedicated to developing pipelines for the physical characterization and synthetic generation of supernova light curves. The project utilizes Machine Learning approaches (MLP, Bayesian Networks, Random Forest) to map the relationship between stellar physical parameters and photometric observations, including realistic simulations of LSST sampling rates.

# Project overview
* Data_Augmentation_&_Sampling: Tools for data corruption, noise injection, and non-uniform sampling simulation based on Vera C. Rubin Observatory (LSST) specifications.

* Physics_Characterization: Regression models (Curves → Parameters) to estimate physical properties such as radius, mass, and energy from light curves.

* LightCurve_Generation: Generative models (Parameters → Curves) to synthesize new light curves based on physical inputs.

* Unified_Frameworks: Integrated scripts that simultaneously perform characterization and generation tasks for benchmarking different algorithms.

# Repository structure
```
ASTRAI/
├── 4_par_synth_dataset/               # 4-parameter synthetic dataset and light curves
├── 4_par_models/                      # Training frameworks & models for 4-parameter data
├── 7_par_dataset/                     # 7-parameter synthetic dataset and light curves
├── 7_par_models/                      # Core models and training frameworks (7 parameters)
│   ├── bayesian_network_unified_7_par/ # Bayesian Ridge Regression pipeline
│   ├── full_model_sup_unp_7_par/       # Deep Learning (Split-MLP) & ResNet architectures
│   └── random_forest_unified_7_par/    # Random Forest Regressor framework
├── observations/                       # Collected observational data
├── real_dataset/                       # Real-world dataset for validation/testing
├── venv/                               # Python virtual environment (ignored by git)
├── LICENSE                             # Project license
└── README.md                           # Project documentation
```

# Core components

## Data sampling and corruption
1. LSST_sampling

    * Simulates realistic observation windows considering solar elevation, lunar luminosity, and cloud cover.

    * Provides algorithms for non-uniform sampling and time-series reconstruction.

2. Data_corruption_v2

    * Manages robust training through Gaussian noise addition and the creation of missing data "patches".

    * Includes fast interpolation methods (Linear or Gaussian Process) to restore curve continuity before processing.

## Machine Learning Models
* Deep Learning (Split-MLP): Implemented in mlpres.py, it uses an independent-head strategy (one network per physical parameter) to maximize characterization accuracy.

* Bayesian Networks: Utilizes BayesianRidge regression to provide probabilistic estimates and manage the intrinsic uncertainty of astronomical data.

* Random Forest: A robust baseline based on decision tree ensembles for CPU performance comparison.

## Performance Evaluation
* All pipelines implement rigorous 10-Fold Cross-Validation.

* Calculated metrics include R2, RMSE, MAE, and RRMSE (Relative Root Mean Squared Error) for standardized evaluation across physical parameters.

# How to navigate
* The *_unified.py files are "turnkey" scripts to reproduce benchmarks for each specific model family.

* For training Deep Learning models, refer to full_model_training.py, which automatically handles data scaling and weight saving.

* Ensure the dataset_preprocessed.csv file is present in the path configured in the scripts or update the BASE_PATH variable accordingly.

# General requirements
* Python 3.10+

* PyTorch (per i modelli MLP)

* scikit-learn, pandas, numpy, matplotlib

* joblib (for Random Forest model serialization)

# Support
For questions or support, contact the development team at Koexai S.r.l.

PNRR Project - Developed as part of the National Recovery and Resilience Plan
