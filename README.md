# Neural Network-derived perfusion maps: a Model-free approach to computed tomography perfusion in patients with acute ischemic stroke

This repository contains the source code for the paper [Neural Network-derived perfusion maps: a Model-free approach to computed tomography perfusion in patients with acute ischemic stroke](https://arxiv.org/pdf/2101.05992.pdf). The dataset UnitoBRAIN is publicly available at the following [link](https://ieee-dataport.org/open-access/unitobrain).

## Referencing
When using this code please reference the main article
```
@article{gava2021neural,
  title={Neural Network-derived perfusion maps: a Model-free approach to computed tomography perfusion in patients with acute ischemic stroke},
  author={Gava, Umberto A and D'Agata, Federico and Tartaglione, Enzo and Grangetto, Marco and Bertolino, Francesca and Santonocito, Ambra and Bennink, Edwin and Bergui, Mauro},
  journal={arXiv preprint arXiv:2101.05992},
  year={2021}
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
* numpy
* torch
* torchvision
* json
* pydicom


## Train model 
```
python3 main.py
``` 
