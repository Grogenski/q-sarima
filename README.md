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
- `qsarima.py`: The core script containing data preprocessing, the PennyLane quantum circuit definition, hybrid objective function, and the walk-forward validation loop.
- `config.yaml`: Configuration file to easily set experiment parameters (variables, climate, horizons, qubits) without modifying the source code.
- `city_climate.json`: Auxiliary mapping file linking dataset files to their respective climate classifications.
- `requirements.txt`: List of Python dependencies required to run the project.

## Requirements
To run the Q-SARIMA pipeline, you need Python 3.8+ and the libraries listed in the `requirements.txt` file. We recommend creating a virtual environment before installing the dependencies:

```
pip install -r requirements.txt
```

## How to Run

-Clone this repository:

```
git clone [https://github.com/Grogenski/q-sarima.git](https://github.com/Grogenski/q-sarima.git)
cd q-sarima
```

-Ensure the `dataset/` folder contains the required JSON climate files.
-(Optional) Adjust the experiment parameters in the `config.yaml` file according to your needs.
-Execute the main script passing the configuration file. The algorithm will automatically read the data, perform the AutoARIMA structural discovery, and run the COBYLA quantum optimization loop:

```
python qsarima.py --config config.yaml
```
_Note: Due to the computational overhead of simulating quantum circuits classically, the process may take some time depending on the number of configured CPU threads._

## Citation
If you use this code or our methodology in your research, please cite our paper:

```
@inproceedings{meloca2026qsarima,
  title={Q-SARIMA: A Hybrid Quantum-Classical Extension of SARIMA for Time Series Forecasting in Precision Agriculture},
  author={Meloca, Lucas Grogenski and de Souza, Rodrigo Clemente Thom and Aylon, Linnyer Beatrys Ruiz and Mariani, Viviana Cocco},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2026},
  organization={IEEE}
}
```

## Acknowledgments
This work was supported by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 001, the Manna Team, Softex, Fundação Araucária, and the Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq) under grant 402015/2025-8 (Call CNPq/MCTI No. 44/2024 - Faixa B).

## License
This project is licensed under the MIT License - see the LICENSE file for details.


