"""
Configuration file for the project.
Contains paths, default hyperparameters, and constants.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Data paths
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")

# Model paths
LOGREG_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm")
