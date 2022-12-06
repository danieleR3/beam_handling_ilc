# ilc_flexible_beam
# ILC for Vibration Free Flexible Object Handling with a Robot Manipulator

## Introduction
This repository provides an iterative learning control approach that jointly learns model parameters and residual dynamics of a robot manipulator handling a flexible object by means of the interoceptive sensors of the robot (motor encoder and motor torques). The learned model is utilized to design optimal (PTP) trajectories that accounts for residual vibration, nonlinear kinematics of the manipulator and joint limits.

## Getting started
1. Install required packges listed in `requirements.txt` (`beam_handling_ilc_py` folder)
2. Install python package by running `pip install -e .` from `beam_handling_ilc_py` folder
3. Build beam handling catkin workspace (`beam_handling_ilc_ws`) to use RViz for visualizing the trajectory of the robot
4. Run `ilc_optimal_control.py`, `ilc_estimator.py`

The problem is described in detail in the preprint arXiv paper [arxiv.2211.11076](https://doi.org/10.48550/arXiv.2211.11076
) that have been submitted for the IFAC World Congress 2023. 

## Citing
To cite our paper in your academic research, please use the following bibtex line:
```bibtex
@article{Ronzani2022,
archivePrefix = {arXiv},
arxivId = {2211.11076},
author = {Ronzani, Daniele and Mamedov, Shamil and Swevers, Jan},
doi = {10.48550/arxiv.2211.11076},
eprint = {2211.11076},
month = {nov},
title = {{Vibration Free Flexible Object Handling with a Robot Manipulator Using Learning Control}},
url = {https://arxiv.org/abs/2211.11076v1},
year = {2022}
}

```