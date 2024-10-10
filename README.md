# UQ_Netperf

## Project Overview
This project focuses on the **Uncertainty Quantification (UQ)** of network throughput predictions, a crucial aspect of ensuring efficient data transfer for large-scale scientific discoveries. Scientific user facilities generate massive datasets, and remote data transfers play a key role in fields like climate modeling, bioinformatics, and particle physics. Effective UQ can help allocate network and storage resources more efficiently, improving performance and reliability.

The project aims to develop and evaluate UQ methods, including **Bootstrap**, **Conformal Prediction**, and **Predictability, Computability, and Stability (PCS)**, to provide accurate perturbation intervals for predicting future data transfers.

## Goals
### Achievable:
- Implement UQ techniques on networking **Tstat data**.
- Compare UQ methods in terms of coverage, interval width, and computational efficiency.

### Stretch Goals:
- Develop adaptive UQ methods that respond to real-time network conditions.
- Publish research in a **Network Performance** conference.

## Methods & Data
- **UQ Techniques**: Bootstrap, Conformal Prediction, PCS Framework.
- **Machine Learning**: Regression models such as Random Forest and Conditional RNN for throughput prediction.
- **Evaluation**: Coverage probability, interval width, prediction accuracy.
  
### Dataset
The dataset will be sourced from **Tstat logs** at NERSC (National Energy Research Scientific Computing Center), providing detailed information on network transfers, including throughput, packet loss, and other characteristics.
