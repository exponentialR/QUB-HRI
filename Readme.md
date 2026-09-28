# Repository of Preprocessing of QUB-Perception of Human Engagement in assembly Operations Dataset (QUB-PHEO V1.0) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13956098.svg)](https://doi.org/10.5281/zenodo.13956098)
![GitHub](https://img.shields.io/github/license/exponentialR/QUB-HRI)

## Introduction
<!-- Embed the GIF -->
![QUB-PHEO-Overview](media/qub-pheo.gif)
## Description
One of the core stages of efficient human-robot collaboration (HRC) is human-intention inference, enabling robots to anticipate and respond to human actions seamlessly. Existing approaches often rely on rule-based models or handcrafted heuristics, which lack adaptability to dynamic environments. In contrast, learning-based approaches leverage data-driven models to infer human intent, but their effectiveness depends on the availability of high-quality, multi-view datasets that capture rich spatial-temporal cues.
To address this, we introduce QUB-PHEO, a novel visual-based dyadic multi-view dataset designed to enhance intention inference in HRC. The dataset consists of synchronized multi-view recordings of 70 participants performing 36 distinct assembly subtasks, providing fine-grained labels for action recognition, gaze estimation, and object tracking. By enabling deep learning models to learn intent prediction from diverse viewpoints, QUB-PHEO paves the way for proactive and adaptive robotic collaboration in real-world settings.

QUB-PHEO and its methodology were published in [*IEEE Access* in 2024](https://doi.org/10.1109/ACCESS.2024.3485162).

## Dataset


## Visualiser

The repository includes a local browser viewer for synchronized **AV, UL, UR, LL and LR** videos and saved landmarks, including aerial gaze and object boxes. It works with a local dataset copy and supports any subset of these views. No GPU or model checkpoints are needed. The [visualiser guide](docs/visualiser.md#normalize-and-import-historical-lower-views) also explains how to normalize and import historical LL/LR landmarks while preserving their original pixel arrays.

Use **Python 3.10–3.12** and install **FFmpeg** (`ffprobe` must be on `PATH`). From the cloned repository:

```bash
python3 -m venv .venv-viewer
.venv-viewer/bin/python -m pip install -r requirements-viewer.txt
cp .env.example .env
```

Set `QUB_PHEO_DATASET_ROOT="/path/to/dataset"` in `.env`. That directory should contain `videos/<task>/*.mp4` and matching `landmarks/<task>/*.h5`. Then run:

```bash
.venv-viewer/bin/python visualise.py
```

Open **http://127.0.0.1:8767/**. Your `.env` stays local and is ignored by Git. Dataset files are read only and are supplied separately from the repository.

See [the visualiser guide](docs/visualiser.md) for system prerequisites, Windows instructions, supported formats, configuration overrides and troubleshooting. Use `visualise.py --check` to check dataset discovery without starting the server.

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
For the QUB-PHEO dataset and its published methodology, cite the paper:

```bibtex
@article{adebayo2024qubpheo,
  author  = {Samuel Adebayo and Se{\'a}n McLoone and Joost C. Dessing},
  title   = {{QUB-PHEO}: A Visual-Based Dyadic Multi-View Dataset for Intention Inference in Collaborative Assembly},
  journal = {IEEE Access},
  volume  = {12},
  pages   = {157050--157066},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3485162}
}
```

The preprocessing repository is archived separately on Zenodo as [10.5281/zenodo.13956098](https://doi.org/10.5281/zenodo.13956098):

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
```
