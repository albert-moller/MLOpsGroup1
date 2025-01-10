# MLOps Project Description
#### Goal
This is the project description for the 02476 Machine Learning Operations course at DTU. The aim of this project is to apply course material to a machine learning problem, specifically detecting plant diseases through image classification. The project will leverage pre-trained models and advanced frameworks to achieve accurate predictions. Findings will be presented and submitted alongside the written code.

#### Framework
The selected framework for this project is PyTorch Image Models (TIMM), which provides a variety of pre-trained models for efficient transfer learning. Albumentations will be used for advanced image augmentations, enhancing the training process. These frameworks will be integrated into the project environment while adhering to structured coding practices with version control.

#### Data 
The dataset chosen is the PlantVillage dataset, sourced from Kaggle. It contains over 50,000 images of healthy and diseased plant leaves, spanning multiple plant species and disease types. This dataset will be used to train and validate the classification model.

#### Models
The project will parameter-efficient fine-tune a pre-trained image classification model from the TIMM framework. The chosen architecture is MobileNetV3. The model will be evaluated for its ability to accurately classify plant diseases, leveraging transfer learning to optimize performance.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).


---

# TODO: Week 1

- [x] Albert: Create a git repository (M5)
- [x] Albert: Make sure that all team members have write access to the GitHub repository (M5)
- [x] Everyone: Create a dedicated environment for you project to keep track of your packages (M2)
- [x] Mark: Create the initial file structure using cookiecutter with an appropriate template (M6)
- [x] Cathialina: Fill out the data.py file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
- [x] Albert: Add a model to model.py and a training procedure to train.py and get that running (M6)
- [x] Albert: Remember to fill out the requirements.txtand requirements_dev.txt file with whatever dependencies that you are using (M2+M6)
- [x] Everyone: Remember to comply with good coding practices (pep8) while doing the project (M7)
- [ ] Mark: Do a bit of code typing and remember to document essential parts of your code (M7)
- [ ] Mark: Setup version control for your data or part of your data (M8)
- [ ] Cathialina: Add command line interfaces and project commands to your code where it makes sense (M9)
- [ ] Mark: Construct one or multiple docker files for your code (M10)
- [ ] Mark: Build the docker files locally and make sure they work as intended (M10) 
- [ ] Albert: Write one or multiple configurations files for your experiments (M11)
- [ ] Albert: Used Hydra to load the configurations and manage your hyperparameters (M11)
- [ ] Cathialina: Use profiling to optimize your code (M12)
- [x] Mark: Use logging to log important events in your code (M14)
- [x] Mark: Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
- [ ] Albert: Consider running a hyperparameter optimization sweep (M14)
- [ ] Cathialina: Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)