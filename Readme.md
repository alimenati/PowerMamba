# <center>PowerMamba</center>

This repository contains the code and resources for the research project *"PowerMamba: A Deep State Space Model and Comprehensive Benchmark for Time Series Prediction in Electric Power Systems."*

Our main proposal and the performance are shown in the next two figures:

<div style="text-align: center;">
    <img src="pics/PowerMamba_arc.png" alt="PowerMamba Model">
</div>
<img src="pics/without_pred.png" alt="Prediction results without external forecasts">
<img src="pics/With_pred.png" alt="Comparing prediction results with and without external forecasts">
<div style="text-align: center; margin-top: 20px;">
    <img src="pics/parameters.png" alt="at">
    <img src="pics/context.png" alt="at">
</div>



## Getting Started

To set up the required environment, follow these steps:

```bash
conda env create -f environment.yml
conda activate mamba4ts
```

## Repository Overview

- **`PowerMamba`**: Contains the implementation of the proposed PowerMamba model along with baseline models.
- **`data`**: Includes the benchmark dataset used in this project.

Each folder contains a `README` file with more details about its contents.

# Maintainers
* [Ali Menati](https://scholar.google.com/citations?user=HPreuloAAAAJ&hl=en&oi=ao)
* [Fatemeh Doudi](https://fatemehdoudi.github.io/)


