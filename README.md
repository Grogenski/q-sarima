# q-sarima
Official repository for "Q-SARIMA: A Hybrid Quantum-Classical Extension of SARIMA for Time Series Forecasting in Precision Agriculture" (IJCNN 2026). It utilizes PennyLane and Variational Quantum Circuits (VQC) to optimize classical statistical models for agrometeorological forecasting.

# Q-SARIMA: A Hybrid Quantum-Classical Extension of SARIMA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-purple.svg)](https://pennylane.ai/)

This repository contains the official implementation of the paper **"Q-SARIMA: A Hybrid Quantum-Classical Extension of SARIMA for Time Series Forecasting in Precision Agriculture"**, accepted at the International Joint Conference on Neural Networks (IJCNN).

## Overview
Q-SARIMA is a hybrid framework that integrates classical SARIMA structural identification with Variational Quantum Circuits (VQC) for parameter optimization. Designed to operate within the constraints of the Noisy Intermediate-Scale Quantum (NISQ) era, this model maps the parameter space of classical time series models to the Pauli-Z expectation values of a VQC, functioning as a heuristic regularization mechanism.

This code was developed to forecast 10-day (decendial) agrometeorological time series (Temperature, Relative Humidity, Wind Speed, and Solar Radiation) across distinct Brazilian climate regimes (Tropical Monsoon and Tropical Savanna).

## Project Structure
- `dataset/`: Contains the JSON files with meteorological data sourced from NASA/POWER.
- `results/`: Directory where the raw predictions and final evaluated metrics (RMSE, R² adj., MBE) are saved as CSV files.
- `main.py`: The core script containing data preprocessing, the PennyLane quantum circuit definition, hybrid objective function, and the walk-forward validation loop.

## Requirements
To run the Q-SARIMA pipeline, you need Python 3.8+ and the following libraries:

```bash
pip install numpy pandas scipy tqdm joblib pmdarima pennylane
