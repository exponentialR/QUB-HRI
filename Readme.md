# Repository of Preprocessing of QUB-Perception of Human Engagement in assembly Operations Dataset (QUB-PHEO V1.0) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13956098.svg)](https://doi.org/10.5281/zenodo.13956098)
![GitHub](https://img.shields.io/github/license/exponentialR/QUB-HRI)

## Introduction
<!-- Embed the GIF -->
![QUB-PHEO-Overview](media/qub-pheo.gif)
## Description
One of the core stages of efficient human-robot collaboration (HRC) is human-intention inference, enabling robots to anticipate and respond to human actions seamlessly. Existing approaches often rely on rule-based models or handcrafted heuristics, which lack adaptability to dynamic environments. In contrast, learning-based approaches leverage data-driven models to infer human intent, but their effectiveness depends on the availability of high-quality, multi-view datasets that capture rich spatial-temporal cues.
To address this, we introduce QUB-PHEO, a novel visual-based dyadic multi-view dataset designed to enhance intention inference in HRC. The dataset consists of synchronized multi-view recordings of 70 participants performing 36 distinct assembly subtasks, providing fine-grained labels for action recognition, gaze estimation, and object tracking. By enabling deep learning models to learn intent prediction from diverse viewpoints, QUB-PHEO paves the way for proactive and adaptive robotic collaboration in real-world settings.
## Dataset


## Preprocessing

## Eula and License
To get access to the dataset, please download and fill out the [End User License Agreement](https://github.com/exponentialR/QUB-HRI/license/EULA.md) and send it to [Samuel Adebayo](mailto:samueladebayo@ieee.org)
In using this dataset, you agree to the terms of the license described in the LICENSE file included in this repository.

## What is in the Dataset
- The dataset contains the following:
  - `Annotations` folder: This folder contains the annotations for the dataset. The annotations are in the form of hdf5 files.
  - `Videos` folder: This folder contains the videos for the dataset. The videos are in the form of mp4 files.
  - `README.md` file: This file contains the description of the dataset.
  - `LICENSE` file: This file contains the license for the dataset.
  - `EULA` file: This file contains the End User License Agreement for the dataset.


## Citation
If you use this code for your research, please cite our paper.
```bibtex

@misc{adebayo_exponentialrqub-hri_2024,
	title = {{exponentialR}/{QUB}-{HRI}: v1.1},
	shorttitle = {{exponentialR}/{QUB}-{HRI}},
	url = {https://zenodo.org/records/13956098},
	abstract = {Preprocessing Repository of QUB-Perception of Human Enagagement in Assembly Operations Dataset},
	urldate = {2024-10-19},
	publisher = {Zenodo},
	author = {Adebayo, Samuel},
	month = oct,
	year = {2024},
	doi = {10.5281/zenodo.13956098},
}
