# Hunger increases delay discounting of food and non-food rewards

This repository contains data and analysis code for the paper:

> Skrynka, J., & Vincent, B. T. (2019). Hunger increases delay discounting of food and non-food rewards. _Psychonomic Bulletin and Review_ [https://doi.org/10.3758/s13423-019-01655-0](https://doi.org/10.3758/s13423-019-01655-0)

This paper is open access, but we provide a pre-print of the author accepted manuscript on [PsyArXiv](https://psyarxiv.com/qgp54)

## Data

- `data/discounting/` files in this folder correspond to the raw delay discounting choice data
- `data/data.csv` contains participant data for both conditions, including the subjective hunger measures.

## Analyses

The analyses were conducted in Python and are presented in the form of a number of Jupyter notebooks in the `analysis` folder. These can be viewed online (either on the OSF or in GitHub).

1. `analysis/01_subjective_hunger.ipynb` Analysis of subjective hunger
2. `analysis/02_score_discounting_data.ipynb` Bayesian scoring of raw discounting data
3. `analysis/03_analyse-hyperbolic.ipynb` Analysis of hyperbolic discount function
4. `analysis/04_analyse_hypotheses_hyperbolic_logk.ipynb` Evaluate hypotheses based on hyperbolic discount function
5. `analysis/05_analyse_AUC.ipynb` Analysis of AUC from multiple discount functions
6. `analysis/06_model_comparison.ipynb` Comparison of different discount functions
7. `Evaluate hypotheses based on AUC from multiple discount functions` Evaluate hypotheses based on AUC from multiple discount functions

Running these notebooks will produce a series of outputs which are also contained in the `analysis` folder. These outputs are primarily generated figures or generated data stored in `.csv` files.

There is also a `.jasp` file which includes Bayesian repeated measures t-tests. This filetype should be viewable online, but can also be viewed and explored in the JASP software available from https://jasp-stats.org.

## Packages used
We used the following Python packages

- pandas
- numpy
- matplotlib
- scipy
- dabest: https://github.com/ACCLAB/DABEST-python
- seaborn
- PyMC3: https://github.com/pymc-devs/pymc3
