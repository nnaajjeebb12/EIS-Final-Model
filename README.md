# Battery State Classification Model

This project implements a machine learning model to classify battery states into three categories: New, Used, and Degraded, based on electrochemical impedance spectroscopy (EIS) data.

## Overview

The model analyzes battery impedance data from Excel files containing frequency, real, and imaginary components to determine the battery's state. It uses a combination of feature engineering and deep learning to make predictions.

## Project Structure

```
├── battery_classifier.keras    # Trained neural network model
├── Final_Model.py             # Main prediction script
├── model_selector.pkl         # Feature selector model
├── scaler_battery.pkl         # Feature scaler model
├── Evaluation_Data/
│   ├── degraded/             # Test data for degraded batteries
│   ├── new/                  # Test data for new batteries
│   └── used/                 # Test data for used batteries
└── Training_Data/
    └── DATA_MODEL.xlsx       # Training dataset
```

## Features

- Battery state classification into three categories (New, Used, Degraded)
- Feature engineering from EIS data including:
  - Impedance magnitude and phase angle
  - Admittance calculations
  - Frequency normalization
  - Statistical features (mean, std, min, max)
  - Frequency-domain analysis
- Detailed evaluation reporting with confidence scores
- Summary statistics for model performance

## Installation

1. Clone this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your data follows the required format (Excel files with Frequency, Real, Imaginary columns)
2. Place test files in appropriate folders under `Evaluation_Data/`
3. Run the model:

```bash
python Final_Model_Graph.py
```

## Data Format

Input Excel files should contain the following columns:

- Frequency: Measurement frequency in Hz
- Real: Real component of impedance
- Imaginary: Imaginary component of impedance

## Output

The script provides detailed classification results including:

- Individual predictions with confidence scores
- Per-category accuracy statistics
- Overall model performance metrics
- Prediction distribution breakdown
- Bode plot Vectors on individual points (1 battery)

## Model Architecture

The model uses a neural network architecture with feature selection and scaling:

- Preprocessed features from EIS data
- Scaled and selected features using scikit-learn
- Neural network classifier with TensorFlow/Keras
- Three-class output (New, Used, Degraded)
