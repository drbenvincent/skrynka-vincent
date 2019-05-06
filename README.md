# Hunger increases delay discounting of food and non-food rewards

This repository contains data and analysis code for the paper:

> Skrynka, J., & Vincent, B. T. (2017, July 26). Hunger increases delay discounting of food and non-food rewards. https://doi.org/10.31234/osf.io/qgp54

**STATUS: Under review**


## Data

- `data/discounting/` files in this folder correspond to the raw delay discounting choice data
- `data/data.csv` contains participant data for both conditions, including the subjective hunger measures.


## Analyses

The analyses were conducted in Python and are presented in the form of a number of Jupyter notebooks in the `analysis` folder. These can be viewed online (either on the OSF or in GitHub).

_Note these links work when viewed on GitHub_

1. [Analysis of subjective hunger](analysis/01_subjective_hunger.ipynb)
2. [Bayesian scoring of raw discounting data](analysis/02_score_discounting_data.ipynb)
3. [Analysis of hyperbolic discount function](analysis/03_analyse-hyperbolic.ipynb)
4. [Evaluate hypotheses based on hyperbolic discount function](analysis/04_analyse_hypotheses_hyperbolic_logk.ipynb)
5. [Analysis of AUC from multiple discount functions](analysis/05_analyse_AUC.ipynb)
6. [Comparison of different discount functions](analysis/06_model_comparison.ipynb)
7. [Evaluate hypotheses based on AUC from multiple discount functions](analysis/07_analyse_hypotheses_AUC.ipynb)

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
