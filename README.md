# BeanGuard – Bean Leaf Disease Detection and Monitoring System

## 1. Project Overview

BeanGuard is a machine-learning-based application designed to detect diseases in bean leaves using image classification. It helps users identify three bean leaf conditions:

- Angular Leaf Spot
- Bean Rust
- Healthy

The system uses a MobileNetV2 deep-learning model to classify uploaded bean leaf images. It provides a REST API for image predictions and a web dashboard for monitoring the model, viewing insights, uploading labelled images, and triggering model retraining.

BeanGuard was developed as a machine-learning and MLOps project, combining model development, API development, deployment, monitoring, automated testing, and performance evaluation.

**Important:** The current model is designed specifically for bean leaves. It does not claim to diagnose diseases in other crops.

## 2. Project Objectives

The main objectives of BeanGuard are to:

- Develop a machine-learning model that identifies bean leaf diseases.
- Train and evaluate the model using labelled bean leaf images.
- Provide an API that accepts images and returns predictions.
- Build a dashboard for interacting with and monitoring the system.
- Support uploading new labelled images for future model improvement.
- Implement a retraining mechanism that can update the deployed model.
- Deploy the application to the cloud.
- Test API reliability and performance under multiple user requests.
- Apply software engineering and MLOps practices throughout the project.

## 3. Disease Classes

The model recognizes three classes:

| Class | Description |
|---|---|
| `angular_leaf_spot` | Bean leaves affected by angular leaf spot disease. |
| `bean_rust` | Bean leaves affected by bean rust disease. |
| `healthy` | Bean leaves without the two targeted diseases. |

The model returns a predicted class and a confidence score for an uploaded image.

## 4. Technologies Used

### Machine Learning
- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Image classification

### Backend
- FastAPI
- Uvicorn
- Pydantic

### Dashboard
- Streamlit
- Requests
- Data visualization tools

### DevOps and Deployment
- Docker
- Docker Compose
- Nginx
- Render
- Git and GitHub

### Testing and Performance
- Pytest
- Locust
- Flake8

### Development Environment
- Python virtual environment (`.venv`)
- Visual Studio Code
- PowerShell

## 5. Dataset

BeanGuard uses a labelled image dataset organized into training and testing directories.

The dataset contains three categories:

- `angular_leaf_spot`
- `bean_rust`
- `healthy`

The training dataset is used to teach the model to recognize the different bean leaf conditions. The test dataset is used to evaluate the model on images that were not used for training.

The local dataset currently contains:

- **1,169 image files**
- Approximately **120.8 MB** of image data across the training and testing directories.

The project also includes a location for newly uploaded labelled images.

### Dataset Structure

```text
data/
├── train/
│   ├── angular_leaf_spot/
│   ├── bean_rust/
│   └── healthy/
│
├── test/
│   ├── angular_leaf_spot/
│   ├── bean_rust/
│   └── healthy/
│
├── new_data_demo/
│
└── uploads/
```

The `uploads` directory is intended for newly submitted images. Uploaded images can be used as additional labelled examples during the retraining process.

## 6. Machine Learning Model

BeanGuard uses MobileNetV2 for image classification.

MobileNetV2 is a convolutional neural network architecture designed to perform image recognition efficiently. It is suitable for applications that need image classification without relying on an unnecessarily large model.

The model predicts one of the three supported bean leaf classes.

### Model Artifacts

The project includes the following model-related files:

- `beanguard_mobilenetv2.keras` – the saved Keras model.
- `beanguard_mobilenetv2.h5` – an additional saved model format.
- `class_names.json` – stores the class names used by the classifier.
- `metrics.json` – stores model evaluation metrics.

The model is loaded by the API when the application starts.

## 7. System Architecture

BeanGuard follows a simple architecture in which the dashboard communicates with the backend API, and the API uses the trained model to generate predictions.

```text
                 USER
                   |
                   v
          STREAMLIT DASHBOARD
                   |
                   v
              FASTAPI
                   |
          -----------------
          |               |
          v               v
    IMAGE PROCESSING   MONITORING
          |
          v
    MOBILENETV2 MODEL
          |
          v
    PREDICTED CLASS
          |
          v
    RESULT TO DASHBOARD
```

The system also supports a retraining workflow:

```text
New Labelled Images
        |
        v
Upload Endpoint
        |
        v
Pending Uploads
        |
        v
Retraining Trigger
        |
        v
Training and Evaluation
        |
        v
Candidate Model
        |
        v
Model Evaluation
        |
        v
Model Promotion
```

The retraining process is designed to evaluate a candidate model before replacing the existing model.

## 8. Backend API

The backend is developed using FastAPI.

It provides endpoints for health checks, model information, predictions, image uploads, monitoring, and retraining.

### Main API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Checks whether the API is healthy. |
| GET | `/status` | Returns model and application status. |
| GET | `/metrics` | Provides model and system metrics. |
| GET | `/predictions/recent` | Retrieves recent prediction records. |
| POST | `/predict` | Accepts an image and returns a prediction. |
| POST | `/upload` | Uploads labelled images for future training. |
| POST | `/retrain` | Starts the model retraining process. |
| GET | `/retrain/status` | Checks the retraining process status. |
| GET | `/insights` | Provides data and prediction insights. |

### Example Prediction Response

A successful prediction returns information about the detected class and its confidence score.

The response structure depends on the API implementation, but it includes the predicted disease category and classification confidence.

### API Documentation

FastAPI automatically provides interactive API documentation.

Visit:

https://beanguard.onrender.com/docs

## 9. Streamlit Dashboard

BeanGuard includes a Streamlit dashboard that provides a user-friendly interface for interacting with the deployed API.

The dashboard supports the following activities:

- Viewing application and API status.
- Checking whether the machine-learning model is available.
- Uploading bean leaf images for diagnosis.
- Viewing prediction results.
- Exploring available data insights.
- Uploading labelled images.
- Triggering model retraining.
- Monitoring retraining progress.

The dashboard communicates with the backend API rather than independently running the prediction model.

## 10. Model Retraining

BeanGuard includes a retraining workflow that allows new labelled images to contribute to future model updates.

The process works as follows:

1. A user uploads labelled bean leaf images.
2. The API accepts the images and stores them as pending uploads.
3. The retraining endpoint is triggered.
4. The retraining process checks for available training images.
5. The candidate model is trained and evaluated.
6. The candidate is compared with the existing model.
7. The model is promoted only when it satisfies the configured evaluation conditions.
8. The retraining status is updated.

The retraining implementation also checks that the expected class directories exist.

### Retraining Endpoints

Start retraining:

```http
POST /retrain
```

Check retraining status:

```http
GET /retrain/status
```

A response indicating that retraining has started does not necessarily mean it has completed successfully. The status endpoint must be checked to confirm the final result.

### Current Deployment Consideration

During cloud testing, retraining failed because the deployed API could not find the original training images at:

```text
/app/data/train
```

The local training and testing datasets exist, but they were not initially tracked in Git. The Docker configuration was updated to include the training and testing directories.

The dataset files still need to be confirmed in the GitHub repository and included in a successful Render deployment before cloud retraining can be considered fully verified.

## 11. Deployment

BeanGuard has been deployed using Render.

The project has two deployed services:

**Backend API**

https://beanguard.onrender.com

**Streamlit Dashboard**

https://beanguard-dashboard.onrender.com

### Useful Deployment Links

- API health: https://beanguard.onrender.com/health
- Application status: https://beanguard.onrender.com/status
- API documentation: https://beanguard.onrender.com/docs
- Retraining status: https://beanguard.onrender.com/retrain/status

The backend and dashboard are separate services. The dashboard sends requests to the backend API.

### Docker

Docker was used to containerize the backend and support consistent application deployment.

The API Docker image was built successfully using:

```dockerfile
FROM python:3.11-slim
```

The API container uses Uvicorn to serve the FastAPI application.

The Docker configuration includes the application source code, model files, dependencies, and the intended training and testing dataset directories.

A health check is also configured to check the API's `/health` endpoint.

### Local Docker and Scaling Tests

The project was also tested locally using Docker containers.

The API, dashboard, and Nginx containers were running during the local deployment tests.

The API was tested with one, two, and three containers behind Nginx. Two-container and three-container configurations were confirmed to be healthy.

## 12. Testing

Testing was performed to check whether the application works correctly and whether its main components behave as expected.

### Unit and Application Tests

Pytest was used to run the project's automated tests.

The recorded test result was:

```text
12 passed, 4 warnings
```

This confirmed that the 12 tests passed at the time of testing. The warnings did not prevent the tests from passing.

### API Testing

The API was tested using its endpoints.

The following functionality was checked:

- API health checks.
- Model availability.
- Image prediction.
- Image uploads.
- Retraining requests.
- Retraining status monitoring.

The prediction endpoint successfully returned a classification result during testing. One recorded angular leaf spot prediction had a confidence score of 92.76%.

Image uploads were also accepted successfully by the API.

## 13. Performance and Load Testing

Locust was used to simulate multiple users sending requests to the API.

The load testing focused on:

- `/health`
- `/predict`

### Recorded One-Container Test Results

| Metric | `/health` | `/predict` |
|---|---:|---:|
| Total requests | 38 | 159 |
| Failures | 0 | 0 |
| Average response time | 965.68 ms | 11,261.45 ms |
| Median response time | Not recorded | 12,000 ms |
| 95th percentile | Not recorded | 23,000 ms |

**Overall results:**

- Total requests: 197
- Total failures: 0
- Average response time across requests: 9,275.47 ms

The results showed that the API handled the recorded requests without failures. However, image prediction took considerably longer than health checks.

The project also tested two and three API containers behind Nginx. The corresponding detailed performance results were not recorded here.

## 14. Continuous Integration

GitHub Actions was configured to run automated checks when changes are pushed to the repository.

The CI workflow uses Python 3.11 and includes:

- Installing project dependencies.
- Running Flake8 for code quality checks.
- Running Pytest for automated tests.

The CI workflow passed during the recorded testing.

## 15. Running the Project Locally

### Requirements

Install the following before running the project:

- Python 3.11
- Git
- Docker (optional, for containerized execution)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Shema-8/BeanGuard.git
```

### Step 2: Enter the Project Directory

```bash
cd BeanGuard
```

### Step 3: Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 6: Start the API

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API should be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

The Streamlit dashboard can be started separately, using the project's dashboard entry point:

```powershell
streamlit run ui/app.py
```

The dashboard must be configured to communicate with the API URL being used.

## 16. Running Automated Tests

Run the test suite using:

```powershell
pytest -q
```

Run the code quality check using:

```powershell
flake8
```

## 17. Running Load Tests

Install Locust if it is not already installed:

```powershell
pip install locust
```

Start Locust using the project's Locust configuration file:

```powershell
locust -f locust/locustfile.py
```

Open the Locust interface in a browser using the local address displayed by Locust. During our local testing, the interface was accessed through port `8090`.

Configure the target API host and the number of simulated users before starting the test.

## 18. Known Limitations and Considerations

- The model currently supports only three bean leaf categories.
- Prediction response times can be relatively high, especially under load.
- Render free services can sleep after periods of inactivity, which can cause slower responses when they wake up.
- Local container filesystems and cloud container filesystems may be temporary. Uploaded images should not be assumed to persist after a service restart unless persistent storage is configured.
- Cloud retraining requires the original training and testing datasets to be available inside the deployed container.
- A successful retraining trigger does not guarantee successful model training. The retraining status must be checked.
- The cloud retraining workflow still requires a successful end-to-end verification after the dataset deployment issue is resolved.

## 19. Future Improvements

Possible improvements include:

- Expanding the dataset with more diverse bean leaf images.
- Improving model prediction speed.
- Increasing the number of labelled training examples.
- Improving dashboard visualizations.
- Adding persistent storage for uploaded images.
- Improving model monitoring and retraining logs.
- Performing additional load tests with different numbers of API containers.
- Evaluating the model with additional unseen images.
- Improving deployment reliability and automated model version management.

## 20. Repository

GitHub repository:

https://github.com/Shema-8/BeanGuard

The repository contains the source code, model artifacts, API, dashboard, Docker configuration, tests, and CI workflow.

## 21. Conclusion

BeanGuard demonstrates the development and deployment of an end-to-end machine-learning application for bean leaf disease classification.

The project brings together image classification using MobileNetV2, a FastAPI backend, a Streamlit dashboard, Docker containerization, cloud deployment, automated testing, and load testing.

It also introduces a retraining workflow intended to support future model improvements using newly labelled images.

Through this project, practical experience was gained in machine-learning development, API integration, containerization, deployment, monitoring, testing, and MLOps practices.