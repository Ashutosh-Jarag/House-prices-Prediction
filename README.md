# House Price Prediction

This project is a machine learning application that predicts house prices based on various features of the house. It uses a pipeline-based approach for training and deploying the model, with `zenml` for pipeline management and `mlflow` for experiment tracking.

## Overview

The goal of this project is to build a model that can accurately predict the sale price of a house. The project is structured to facilitate easy experimentation with different models and preprocessing techniques, and to make the deployment of the trained model as smooth as possible.

## Project Structure

The project is organized into the following directories:

- **analysis:** Contains Jupyter notebooks and Python scripts for exploratory data analysis (EDA).
- **data:** Contains the raw data used for training and testing the model.
- **explanations:** Contains files related to design patterns used in the project.
- **extracted_data:** Contains the extracted data from the raw data.
- **mlruns:** Contains the output of MLflow experiment tracking.
- **pipelines:** Contains the `zenml` pipelines for training and deployment.
- **src:** Contains the source code for data ingestion, preprocessing, model training, and evaluation.
- **steps:** Contains the `zenml` steps that are used to build the pipelines.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The project requires Python 3.8 or higher. The dependencies are listed in the `requirements.txt` file.

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline

The project uses `zenml` to define and run the pipelines. There are two main pipelines:

-   **Training Pipeline:** This pipeline ingests the data, preprocesses it, trains a model, and evaluates it. To run the training pipeline, execute the following command:
    ```bash
    python run_pipeline.py
    ```
-   **Deployment Pipeline:** This pipeline deploys the trained model as a web service. To run the deployment pipeline, execute the following command:
    ```bash
    python run_deployment.py
    ```

## Usage

Once the model is deployed, you can use it to make predictions by sending a POST request to the prediction endpoint. The `sample_predict.py` script shows an example of how to do this.

```bash
python sample_predict.py
```

## Experiment Tracking

The project uses `mlflow` to track experiments. You can view the results of your experiments by running the following command:

```bash
mlflow ui
```

This will start the MLflow UI, where you can view the parameters, metrics, and artifacts for each run.
