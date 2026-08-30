# MLOps PyTorch Pipeline

An end-to-end MLOps pipeline for training and serving a PyTorch image classification model using **CIFAR-10**, **ResNet-18**, **Docker**, **GitHub Actions**, and **Kubernetes**.

The project separates model training from model serving, packages both workloads into independent Docker images, stores model checkpoints in persistent storage, and deploys inference as a scalable Kubernetes service.

---

## Architecture

```text
                           ┌──────────────────────┐
                           │     GitHub Repo      │
                           │                      │
                           │ src/                 │
                           │ configs/             │
                           │ docker/              │
                           │ k8s/                 │
                           └──────────┬───────────┘
                                      │
                               GitHub Actions
                                      │
                         Build & Push Docker Images
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │        GHCR          │
                           │ GitHub Container     │
                           │ Registry             │
                           │                      │
                           │ mlops-train:latest   │
                           │ mlops-serve:latest   │
                           └──────────┬───────────┘
                                      │
                                      │ image pull
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         Kubernetes Cluster                                 │
│                                                                            │
│ Namespace: ml-training                                                     │
│                                                                            │
│  ┌──────────────────┐       ┌─────────────────────────────────────────┐    │
│  │    ConfigMap     │──────▶│          Training Job                   │    │
│  │ training-config  │       │                                         │    │
│  │                  │       │ PyTorch + ResNet-18                     │    │
│  │ epochs           │       │ CIFAR-10                                │    │
│  │ batch size       │       │                                         │    │
│  │ learning rate    │       └──────────┬───────────────┬──────────────┘    │
│  └──────────────────┘                  │               │                   │
│                                       │               │                   │
│                                       ▼               ▼                   │
│                              ┌───────────────┐ ┌─────────────────┐         │
│                              │   Data PVC    │ │ Checkpoints PVC │         │
│                              │               │ │                 │         │
│                              │ CIFAR-10      │ │ best_model.pt   │         │
│                              └───────────────┘ └────────┬────────┘         │
│                                                        │ read-only         │
│                                                        ▼                   │
│                                           ┌─────────────────────────┐      │
│                                           │ Serving Deployment      │      │
│                                           │                         │      │
│                                           │      2 replicas         │      │
│                                           │                         │      │
│                                           │ ┌───────┐   ┌───────┐   │      │
│                                           │ │ Pod 1 │   │ Pod 2 │   │      │
│                                           │ │FastAPI│   │FastAPI│   │      │
│                                           │ └───┬───┘   └───┬───┘   │      │
│                                           └─────┼───────────┼───────┘      │
│                                                 │           │              │
│                                                 └─────┬─────┘              │
│                                                       ▼                    │
│                                              ┌─────────────────┐           │
│                                              │ ClusterIP       │           │
│                                              │ Service :80     │           │
│                                              └────────┬────────┘           │
└───────────────────────────────────────────────────────┼────────────────────┘
                                                        │
                                              kubectl port-forward
                                                        │
                                                        ▼
                                             localhost:8080
                                                        │
                                               POST /predict
                                               GET  /health
```

---

## Technology Stack

- Python 3.11
- PyTorch
- Torchvision
- CIFAR-10
- ResNet-18
- FastAPI
- Uvicorn
- Docker
- Docker Compose
- GitHub Actions
- GitHub Container Registry (GHCR)
- Kubernetes
- Kubernetes Persistent Volumes
- Kubernetes ConfigMaps
- Horizontal Pod Autoscaler

---

## Project Structure

```text
mlops-pytorch-pipeline/
├── .github/
│   └── workflows/
│       ├── build-train.yml
│       ├── build-serve.yml
│       ├── copy-project.yml
│       ├── deploy-source.yml
│       ├── pull-image.yml
│       └── sync-compose.yml
│
├── configs/
│   └── training_config.yaml
│
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
│
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
│
├── requirements/
│   ├── train.txt
│   └── serve.txt
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── serve.py
│
├── checkpoints/
├── data/
├── compose.yaml
└── README.md
```

---

# Model

The project uses a modified **ResNet-18** architecture for CIFAR-10 classification.

CIFAR-10 contains 10 classes:

1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

The standard ResNet-18 input layer is adapted for CIFAR-10's `32 × 32` images.

Training uses data augmentation including random horizontal flipping and random cropping.

Images are normalized using CIFAR-10 normalization values.

---

# Training Configuration

Training parameters are stored in:

```text
configs/training_config.yaml
```

Example:

```yaml
model:
  architecture: resnet18
  num_classes: 10

data:
  dataset: CIFAR10
  data_dir: data
  num_workers: 0

training:
  epochs: 20
  batch_size: 64
  learning_rate: 0.001
  early_stopping_patience: 5
  resume: true

output:
  checkpoint_dir: checkpoints
  checkpoint_every: 1
  keep_last_checkpoints: 2
```

The training script supports:

- configurable hyperparameters
- JSON Lines metric logging
- validation
- early stopping
- periodic checkpoints
- best-model checkpointing
- training resume
- final model checkpoint
- training history

---

# Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/rahul-ai-labs/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
```

## 2. Create a Python Environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

## 3. Install Training Dependencies

```bash
pip install -r requirements/train.txt
```

---

# Local Training

Run training from the project root:

```bash
python -m src.train
```

The training configuration is loaded from:

```text
configs/training_config.yaml
```

Training outputs are written to:

```text
checkpoints/
```

Example files:

```text
checkpoints/
├── best_model.pt
├── best_metrics.json
├── final_model.pt
├── final_metrics.json
├── metrics_history.json
└── checkpoint_epoch_*.pt
```

Training metrics are written to stdout as JSON Lines:

```json
{"epoch": 1, "train_loss": 1.1054, "train_accuracy": 0.6213, "val_loss": 0.9024, "val_accuracy": 0.6983}
```

---

# Local Model Serving

Install serving dependencies:

```bash
pip install -r requirements/serve.txt
```

Start FastAPI:

```bash
python -m uvicorn src.serve:app --host 127.0.0.1 --port 8000
```

Check application health:

```bash
curl http://localhost:8000/health
```

Send an image for prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@test_image.jpg"
```

Windows PowerShell:

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -F "image=@test_image.jpg"
```

---

# Docker

The project contains separate Docker images for training and serving.

## Build Training Image

```bash
docker build \
  -f docker/Dockerfile.train \
  -t mlops-train:v1 .
```

Windows PowerShell:

```powershell
docker build -f .\docker\Dockerfile.train -t mlops-train:v1 .
```

## Run Training Container

Linux/macOS:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  mlops-train:v1
```

PowerShell:

```powershell
docker run --rm `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/checkpoints:/app/checkpoints" `
  mlops-train:v1
```

## Build Serving Image

```powershell
docker build -f .\docker\Dockerfile.serve -t mlops-serve:v1 .
```

## Run Serving Container

```powershell
docker run --rm `
  -p 8080:8080 `
  -v "${PWD}/checkpoints:/app/checkpoints" `
  mlops-serve:v1
```

Test:

```powershell
curl.exe http://localhost:8080/health
```

```powershell
curl.exe -X POST http://localhost:8080/predict `
  -F "image=@test_image.jpg"
```

---

# Docker Compose

The project also provides `compose.yaml`.

Start the serving container:

```bash
docker compose up -d serve
```

Stop it:

```bash
docker compose down
```

Run training explicitly:

```bash
docker compose --profile training run --rm train
```

Training is assigned to a Compose profile so that it does not run automatically when the serving stack starts.

---

# Container Registry

Docker images are published to GitHub Container Registry.

Training image:

```text
ghcr.io/rahul-ai-labs/mlops-train:latest
```

Serving image:

```text
ghcr.io/rahul-ai-labs/mlops-serve:latest
```

Images can be pulled with:

```bash
docker pull ghcr.io/rahul-ai-labs/mlops-train:latest
```

```bash
docker pull ghcr.io/rahul-ai-labs/mlops-serve:latest
```

---

# CI/CD with GitHub Actions

GitHub Actions automates Docker image builds and deployment-related operations.

The training image is rebuilt when training Docker dependencies change.

The serving image is rebuilt when serving Docker dependencies change.

Images are pushed to GitHub Container Registry with both:

```text
latest
```

and commit-specific SHA tags.

The workflows also support manual execution using `workflow_dispatch`.

Source-code deployment and image builds are separated to avoid unnecessary Docker image rebuilds.

---

# Kubernetes Deployment

The Kubernetes resources are deployed into:

```text
ml-training
```

namespace.

Verify the cluster first:

```powershell
kubectl cluster-info
kubectl get nodes
```

---

## 1. Create Namespace

```powershell
kubectl apply -f .\k8s\namespace.yaml
```

Verify:

```powershell
kubectl get namespaces
```

---

## 2. Create Persistent Volume Claims

```powershell
kubectl apply -f .\k8s\pvc.yaml
```

Verify:

```powershell
kubectl get pvc -n ml-training
```

The project uses two PVCs:

```text
training-data-pvc
checkpoints-pvc
```

The data PVC stores the dataset, while the checkpoint PVC persists trained model artifacts.

---

## 3. Create Training ConfigMap

```powershell
kubectl apply -f .\k8s\configmap.yaml
```

The ConfigMap exposes the training configuration inside the container at:

```text
/app/configs/training_config.yaml
```

Verify:

```powershell
kubectl get configmap -n ml-training
```

---

# Kubernetes Training Job

Deploy the training Job:

```powershell
kubectl apply -f .\k8s\training-job.yaml
```

Check the Job:

```powershell
kubectl get jobs -n ml-training
```

Check Pods:

```powershell
kubectl get pods -n ml-training
```

Watch the Pod:

```powershell
kubectl get pods -n ml-training -w
```

Stream training logs:

```powershell
kubectl logs -f job/model-training -n ml-training
```

Example:

```json
{"event": "training_started", "device": "cpu"}
{"epoch": 1, "train_loss": 1.1054, "train_accuracy": 0.6213, "val_loss": 0.9024, "val_accuracy": 0.6983}
{"event": "best_model_saved", "epoch": 1, "val_loss": 0.9024, "model_path": "/app/checkpoints/best_model.pt"}
```

The Job mounts:

```text
training-data-pvc  → /app/data
checkpoints-pvc    → /app/checkpoints
training-config    → /app/configs
```

Therefore model checkpoints survive after the training Pod terminates.

---

# GPU Training
In progress, the Kubernetes training manifest in the future can optionally request an NVIDIA GPU:

```yaml
resources:
  limits:
    nvidia.com/gpu: "1"
```

GPU scheduling requires:

- an NVIDIA GPU node
- NVIDIA drivers
- NVIDIA Kubernetes device plugin/runtime configuration
- a CUDA-enabled PyTorch training image

GPU resources should only be requested on a cluster configured for NVIDIA GPU workloads.

---

# Kubernetes Model Serving

After training has completed and `best_model.pt` exists in the checkpoint PVC, deploy the inference application.

```powershell
kubectl apply -f .\k8s\serving-deployment.yaml
```

Deploy the Service:

```powershell
kubectl apply -f .\k8s\serving-service.yaml
```

Deploy the HPA:

```powershell
kubectl apply -f .\k8s\hpa.yaml
```

Check the rollout:

```powershell
kubectl rollout status deployment/model-serving -n ml-training
```

Verify:

```powershell
kubectl get pods -n ml-training
```

The Deployment runs **2 serving replicas**.

---

# Health Checks

The serving Deployment uses Kubernetes liveness and readiness probes against:

```text
GET /health
```

Liveness checks determine whether a container should be restarted.

Readiness checks determine whether a Pod is ready to receive requests through the Kubernetes Service.

Inspect the Deployment with:

```powershell
kubectl describe deployment model-serving -n ml-training
```

---

# Kubernetes Service

The serving application is exposed internally using a `ClusterIP` Service.

```text
Service port:   80
Container port: 8080
```

The request path is:

```text
model-serving Service :80
          ↓
FastAPI Pod :8080
```

Verify:

```powershell
kubectl get svc -n ml-training
```

---

# Testing Kubernetes Inference

Because the Service is `ClusterIP`, use port forwarding for local access:

```powershell
kubectl port-forward svc/model-serving 8080:80 -n ml-training
```

The path becomes:

```text
localhost:8080
      ↓
kubectl port-forward
      ↓
model-serving:80
      ↓
FastAPI Pod:8080
```

Keep the port-forward terminal open.

From another terminal, test health:

```powershell
curl.exe http://localhost:8080/health
```

Test prediction:

```powershell
curl.exe -X POST http://localhost:8080/predict `
  -F "image=@test_image.jpg"
```

---

# Horizontal Pod Autoscaling

The Horizontal Pod Autoscaler scales the model-serving Deployment based on CPU utilization.

```text
k8s/hpa.yaml
```

Check HPA status:

```powershell
kubectl get hpa -n ml-training
```

CPU-based HPA requires Kubernetes Metrics Server.

Check whether metrics are available:

```powershell
kubectl top pods -n ml-training
```

---

# Copying an Existing Model to the Checkpoint PVC

For testing, an existing trained model can be copied to a Pod that mounts `checkpoints-pvc`.

For example:

```powershell
kubectl cp .\checkpoints\best_model.pt `
  ml-training/<pod-name>:/app/checkpoints/best_model.pt
```

To copy the entire checkpoint directory:

```powershell
kubectl cp .\checkpoints\. `
  ml-training/<pod-name>:/app/checkpoints/
```

Verify:

```powershell
kubectl exec <pod-name> -n ml-training -- ls -lah /app/checkpoints
```

Because `/app/checkpoints` is a PVC mount, files written there persist independently of the Pod.

---

# Persistent Storage Architecture

Kubernetes storage follows:

```text
Pod
 │
 │ references
 ▼
PersistentVolumeClaim (PVC)
 │
 │ bound to
 ▼
PersistentVolume (PV)
 │
 ▼
Physical / cluster storage
```

In this project:

```text
Training Pod
     │
     │ write
     ▼
checkpoints-pvc
     │
     │ persistent
     ▼
best_model.pt
     │
     │ read-only
     ▼
Serving Deployment
     │
     ├── FastAPI Pod 1
     └── FastAPI Pod 2
```

This allows the training workload and serving workload to share the trained model without storing it inside the container image.

---

# Useful Kubernetes Commands

View all project resources:

```powershell
kubectl get all -n ml-training
```

View Pods:

```powershell
kubectl get pods -n ml-training
```

View Jobs:

```powershell
kubectl get jobs -n ml-training
```

View PVCs:

```powershell
kubectl get pvc -n ml-training
```

View PersistentVolumes:

```powershell
kubectl get pv
```

View Services:

```powershell
kubectl get svc -n ml-training
```

View logs:

```powershell
kubectl logs -f job/model-training -n ml-training
```

Describe a Pod:

```powershell
kubectl describe pod <pod-name> -n ml-training
```

Describe serving Deployment:

```powershell
kubectl describe deployment model-serving -n ml-training
```

Stop the training Job:

```powershell
kubectl delete job model-training -n ml-training
```

Deleting the Job does not delete the separately managed checkpoint PVC.

---

# End-to-End Validation

The complete Kubernetes workflow is:

```text
1. Create namespace
        ↓
2. Create PVCs
        ↓
3. Create ConfigMap
        ↓
4. Start Training Job
        ↓
5. Train ResNet-18
        ↓
6. Save best_model.pt to checkpoint PVC
        ↓
7. Training Job completes
        ↓
8. Deploy 2 serving replicas
        ↓
9. Mount checkpoint PVC read-only
        ↓
10. Create ClusterIP Service
        ↓
11. Configure HPA
        ↓
12. kubectl port-forward
        ↓
13. POST /predict
```

Commands:

```powershell
kubectl apply -f .\k8s\namespace.yaml
kubectl apply -f .\k8s\pvc.yaml
kubectl apply -f .\k8s\configmap.yaml
kubectl apply -f .\k8s\training-job.yaml
```

After training:

```powershell
kubectl apply -f .\k8s\serving-deployment.yaml
kubectl apply -f .\k8s\serving-service.yaml
kubectl apply -f .\k8s\hpa.yaml
```

Validate:

```powershell
kubectl get pods -n ml-training
kubectl get jobs -n ml-training
kubectl get pvc -n ml-training
kubectl get svc -n ml-training
kubectl describe deployment model-serving -n ml-training
```

Port-forward:

```powershell
kubectl port-forward svc/model-serving 8080:80 -n ml-training
```

Prediction:

```powershell
curl.exe -X POST http://localhost:8080/predict `
  -F "image=@test_image.jpg"
```

---

# Evidence for Final Submission

The final pull request should include terminal output or screenshots demonstrating:

- successful Docker image builds
- training container execution
- JSONL training metrics
- Kubernetes namespace creation
- PVCs in `Bound` state
- ConfigMap creation
- training Job running/completing
- training logs
- checkpoint creation
- two serving Pods running
- successful Deployment rollout
- liveness/readiness configuration
- Service creation
- HPA configuration
- `/health` returning successfully
- `/predict` returning model probabilities

---

## Summary

This project demonstrates a complete ML lifecycle:

**train → checkpoint → package → deploy → serve → health-check → scale**

Training and inference are independently containerized, configuration is externalized through Kubernetes ConfigMaps, model artifacts are persisted using PVCs, and the inference API is deployed as a replicated Kubernetes service.