# Neural Network-derived perfusion maps: a Model-free approach to computed tomography perfusion in patients with acute ischemic stroke

[![DOI](https://zenodo.org/badge/doi/10.3389/fninf.2023.852105.svg)](https://www.frontiersin.org/articles/10.3389/fninf.2023.852105/full)
[![arXiv](https://img.shields.io/badge/arXiv-2101.05992-b31b1b.svg)](https://arxiv.org/abs/2101.05992)

This repository contains the source code for the paper [Neural Network-derived perfusion maps: a Model-free approach to computed tomography perfusion in patients with acute ischemic stroke](https://www.frontiersin.org/articles/10.3389/fninf.2023.852105/full). The dataset UnitoBRAIN is publicly available at the following [link](https://ieee-dataport.org/open-access/unitobrain).

## Referencing
When using this code please reference the main article
```
@ARTICLE{10.3389/fninf.2023.852105,
AUTHOR={Gava, Umberto A. and D’Agata, Federico and Tartaglione, Enzo and Renzulli, Riccardo and Grangetto, Marco and Bertolino, Francesca and Santonocito, Ambra and Bennink, Edwin and Vaudano, Giacomo and Boghi, Andrea and Bergui, Mauro},
TITLE={Neural network-derived perfusion maps: A model-free approach to computed tomography perfusion in patients with acute ischemic stroke},
JOURNAL={Frontiers in Neuroinformatics},
VOLUME={17},
YEAR={2023},
URL={https://www.frontiersin.org/articles/10.3389/fninf.2023.852105},       
DOI={10.3389/fninf.2023.852105},
ISSN={1662-5196}
}
```
and the dataset UnitoBRAIN should be referenced as
```
@data{x8ea-vh16-21,
doi = {10.21227/x8ea-vh16},
url = {https://dx.doi.org/10.21227/x8ea-vh16},
author = {Gava, Umberto and D'Agata, Federico and Bennink, Edwin and Tartaglione, Enzo and Perlo, Daniele and Vernone, Annamaria and Bertolino, Francesca and Ficiarà, Eleonora and Cicerale, Alessandro and Pizzagalli, Fabrizio and Guiot, Caterina and Grangetto, Marco and Bergui, Mauro},
publisher = {IEEE Dataport},
title = {UniTOBrain},
year = {2021} } 
```

## Requirements
To automatically install all the requirements, please run
```
pip3 install -r requirements.txt
```
As an extra requirement, the UnitoBRAIN should be downloaded, from [IEEE Dataport](https://ieee-dataport.org/open-access/unitobrain). 


## Train model 
To train the model, run simply
```
python3 main.py
``` 
Multiple configurations are available - here below a list of the essential subset of them:

* data_path: location of the (already extracted) dataset
* batch_size: minibatch size
* lr: value of the learning rate
