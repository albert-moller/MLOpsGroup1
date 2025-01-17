# Machine Learning Operations (MLops) Project: Group 1

## Project Description

#### Goal
This is the project description for the 02476 Machine Learning Operations course at DTU. The aim of this project is to apply course material, methods and tools to a machine learning problem, specifically detecting plant diseases through image classification. By being able to identify plant diseases, farmers will be able to promptly remove the diseased plants thus reducing their crop losses, while significantly boosting their agricultural productivity and efficiency. The project will leverage pre-trained models and advanced frameworks to achieve accurate predictions. Findings will be presented and submitted alongside the written code.

#### Framework
For this project, we intend to use the TIMM framework for Computer Vision (PyTorch Image Models) as well as the Albumentations framework for image augmentations. We will be using the TIMM framework to construct and load a pre-trained MobileNetV3 deep learning model and fine-tune it using our plant diseases dataset. In addition, we will be using the Albumenations framework for data augmentation to enhance the robustness of the MobileNetV3 model. These frameworks will be integrated into the project environment while adhering to structured coding practices with version control.

#### Data 
The dataset we have chosen to use for our project is the [Plant Village Dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage) from Kaggle. This dataset contains 54,305 images of both diseased and healthy plants collected under controlled conditions. There are a total of 38 classes. This means that the number of images per class is quite sparse. Hence, there is a need to leverage a pre-trained deep-learning model to maximize classification performance. The images will be normalized and various data augmentations will be applied with low probabilities to enhance model robustness. This dataset will be used to train and validate the classification model.

#### Models
The project will parameter-efficient fine-tune a pre-trained image classification model from the TIMM framework. The chosen architecture is [MobileNetV3](https://arxiv.org/abs/1905.02244). The MobileNetV3 model is known for its efficiency and high performance while using minimal computational resources. This will allow us to fine-tune the model on the Plant Village Dataset, while being able to train it on our laptops. The model will be evaluated for its ability to accurately classify plant diseases, leveraging transfer learning to optimize performance.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
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
│   │   ├── config.py 
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
└── visualizations/           # Visualizations
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
- [x] Mark: Do a bit of code typing and remember to document essential parts of your code (M7)
- [x] Mark: Setup version control for your data or part of your data (M8)
- [x] Cathialina: Add command line interfaces and project commands to your code where it makes sense (M9)
- [x] Mark: Construct one or multiple docker files for your code (M10)
- [x] Mark: Build the docker files locally and make sure they work as intended (M10) 
- [x] Albert: Write one or multiple configurations files for your experiments (M11)
- [x] Albert: Used Hydra to load the configurations and manage your hyperparameters (M11)
- [x] Cathialina: Use profiling to optimize your code (M12)
- [x] Mark: Use logging to log important events in your code (M14)
- [x] Mark: Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
- [x] Albert: Consider running a hyperparameter optimization sweep (M14)
- [x] Cathialina: Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

# TODO: Week 2

- [x] Mark: Write unit tests related to the data part of your code (M16)
- [x] Albert: Write unit tests related to model construction and/or model training (M16)
- [ ] Cathialina Calculate the code coverage (M16)
- [x] Mark Get some continuous integration running on the GitHub repository (M17)
- [x] Mark Add caching and multi-OS/Python/PyTorch testing to your continuous integration (M17)
- [x] Albert: Add a linting step to your continuous integration (M17)
- [ ] Cathialina Add pre-commit hooks to your version control setup (M18)
- [x] Albert Add a continuous workflow that triggers when data changes (M19)
- [ ] Cathialina Add a continuous workflow that triggers when changes to the model registry are made (M19)
- [x] Mark Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
- [x] Mark Create a trigger workflow for automatically building your Docker images (M21)
- [x] Albert Get your model training in GCP using either the Engine or Vertex AI (M21)
- [x] Albert: Create a FastAPI application that can do inference using your model (M22)
- [ ] Cathialina Deploy your model in GCP using either Functions or Run as the backend (M23)
- [x] Albert Write API tests for your application and set up continuous integration for these (M24)
- [x] Mark Load test your application (M24)
- [x] Albert Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
- [x] Mark Create a frontend for your API (M26)
