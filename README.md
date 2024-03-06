# MINTY: Rule-based Models that Minimize the Need for Imputing Features with Missing Values

Rule models are often preferred in prediction tasks with tabular inputs as they can be easily interpreted using natural language and provide predictive performance on par with more complex models. However, most rule models’ predictions are undefined or ambiguous when some inputs are missing, forcing users to rely on statistical imputation models or heuristics like zero imputation, undermining the interpretability of the models. In this work, we propose fitting concise yet precise rule models that learn to avoid relying on features with missing values and, therefore, limit their reliance on imputation at test time. We develop MINTY, a method that learns rules in the form of disjunctions between variables that act as replacements for each other when one or more is missing. This results in a sparse linear rule model, regularized to have small dependence on features with missing values, that allows a trade-off between goodness of fit, interpretability, and robustness to missing values at test time. We demonstrate the value of MINTY in experiments using synthetic and real-world data sets and find its predictive performance comparable or favorable to baselines, with smaller reliance on features with missingness.

This repository contains code to run the experiments in [our paper](https://arxiv.org/abs/2311.14108). We show an example using synthesis data but have used several data sets in our paper where you can find detailed descriptions of the preprocessing steps to reproduce the results. 
If you have any questions, please contact [Lena Stempfle ](https://www.chalmers.se/personer/stempfle/)

<p align="center">
  <img src="https://github.com/Healthy-AI/_minty/blob/master/Patient_Intro_Anna.jpg" width="400px" height="250px"/></p>

## Installation 
Get started by cloning the repository and installing the required packages in a new virtual environment. The example follows for OS X. The requirements are saved in a requirement.txt file. 

```bash
$ git clone git@github.com:Healthy-AI/_minty.git
$ cd minty
$ python3 -m venv minty_env
$ source minty_env/bin/activate
$ pip install -r 
```

## Configuration files

For the experiments, there is a config file in [`models`](config) which contains, e.g., data-related paths and parameter grids to sample hyperparameters from. To reproduce the experiments, the configs should only be modified by changing the paths to the data as described below and the results folder.

## Data

To prepare the datasets used in the experiments, run [`data_folder/data.py`](data_folder/datasets.py). The raw data used in each experiment must be downloaded as described below and added to a new folder ['realworld_datasets']


### ADNI
The data for ADNI is obtained from the publicly available [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu) database. ADNI collects clinical data, neuroimaging, and genetic data. The regression task aims to predict the outcome of the ADAS13 (Alzheimer's Disease Assessment Scale) (Mofrad et al, 2021) cognitive test at a 2-year follow-up based on available data at baseline.

### Housing

The Ames housing data set was obtained from [here](https://www.kaggle.com/datasets/lespin/house-prices-dataset) and describes the selling price of individual properties, various features, and details of each home in Ames, Iowa, USA from 2006 to 2010. We selected 51 variables on the quality and quantity of a property's physical attributes, such as measurements of area dimensions for each observation, including the sizes of lots, rooms, porches, and garages or some geographical categorical features related to profiling properties and the neighborhood. In a regression task, we used 1460 observations. 

### Life
The data set related to life expectancy has been collected from the [WHO data repository](https://ourworldindata.org/life-expectancy), and its corresponding economic data was collected from the United Nations website. The data can be publicly accessed through. In a regression task, we aim to predict the life expectancy in years from 193 countries considering data from the years 2000-2025. The final dataset consists of 20 columns and 2864 samples where all predicting variables were then divided into several broad categories: immunization factors, mortality factors, economic factors, and social factors.

## Baselines
We compare our work to five baselines. Install the packages within the virtual environment and download the required git repositories.  

### Lasso, Decision Tree, XGB
For Lasso, Decision Tree, XGB, we used the sklearn implementations. 

### Neumiss
Run in models folder:

```bash
$ git clone https://github.com/marineLM/NeuMiss_sota/tree/master
$ pip install .
```
A detailed installation guide is shared in their git repo. 

### RuleFit
Run in models folder: 
```bash
$ git clone https://github.com/csinva/imodels
$ git install imodels
```
A detailed installation guide is shared in their git repo. Note, that the transform function in rule_fit.py should be changed by: features_r_uses = [term.split(' ')[0] for term in r.split(' and ')] to match with the data generated by the preprocessing steps. 

## Run locally
To run and evaluate a single model, use ['experiment/main.py'](experiment/main.py). Find an overview of all parameter settings in [`models`](config) or the documentation of the corresponding methods, described in the appendix of the paper. 


For one minty model on synthetic data run the following command: 

```bash
$ cd experiment
$ python main.py -ds synth -es minty  -pa optimizer ilp max_rules 7 classifier False feasibility_tol 1e-6 gamma 0.001 lambda_0 0.001 lambda_1 0.001 max_rules  5 optimality_tol 1e-6 reg_fit True reg_rho False relaxed True silent False -i none -sp 0.2 -s 9 -fr 1.0 -it 1
```

## Run on a cluster
Running all experiments is computationally heavy and access to CPUs is required. 
You can use [`experiments/queue_jobs.py`](experiments/queue_jobs.py) to sweep over all models (minty versions and five baselines) and data-set specific hyperparameters and create a results folder. 


## Citation 
If you use this work, please cite it as follows: 
```bib
@article{stempfle2023minty,
      title={MINTY: Rule-based Models that Minimize the Need for Imputing Features with Missing Values}, 
      author={Lena Stempfle and Fredrik D. Johansson},
      journal={arXiv preprint arXiv:2311.14108},
      year={2023},
}
```

## Acknowledgements

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation.

The computations in this work were enabled by resources provided by the Swedish National Infrastructure for Computing (SNIC) at Chalmers Centre for Computational Science and Engineering (C3SE) partially funded by the Swedish Research Council through grant agreement no. 2018-05973.

