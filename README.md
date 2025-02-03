# Strawberry Plant Health AI

A YOLOv8-based AI project for disease detection and classification.

## Overview

This project focuses on leveraging computer vision and YOLOv8 to detect and classify the health of strawberry plants. The workflow includes downloading open dataset, fine-tuning models, real-time classification of images using a camera as well as classification of images on a server.

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Requirements](#requirements)
5. [Setup](#setup)
6. [Usage](#usage)
   - [Run Program](#run-program)
   - [Download Dataset](#download-dataset)
   - [Train Model](#train-model)
   - [Validate Model](#validate-model)
   - [Capture Transform and Classify in Real-Time](#capture-transform-and-classify-in-real-time)
   - [Download Images from Server](#download-images-from-server)
   - [Transform and Classify Server Images](#transform-and-classify-server-images)
   - [Server Operations](#server-operations)
7. [Configuration](#configuration)
8. [API Endpoints](#api-endpoints)
9. [Acknowledgements](#acknowledgements)

## Features
- **Dataset Management**: Download datasets from URLs or a server.
- **Training and Validation**: Fine-tune YOLOv8 on custom datasets.
- **Image Transformation**: Apply transformations to improve detection.
- **Real-Time Classification**: Use a camera for real-time object detection and classification.
- **Server Support**: Upload, list, download, and manage images from a server.

## Project Structure
```
STRAWBERRY_HEALTH_AI/
├── config/                                                 # Project configuration directory
│   ├── config.yaml
├── data/                                                   # Dataset automatically downloaded -> python main.py
│   ├── train/
│   ├── test/
│   ├── valid/
│   ├── data.yaml
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
├── downloaded_images/                                      # Automatically created where downloaded images from server are stored 
├── runs/                                                   # Automatically created where model weights and metrics are stored
│   ├── train/
│   ├── val/
├── src/                                                    # Source code for the package
│   ├── camera_image_capture_track_transform_classify.py
│   ├── download_datasets.py
│   ├── download_images_from_server.py            
│   ├── server_image_transform_classify.py              
│   ├── train_model.py
│   ├── utils.py                
│   ├── validate_model.py
├── transformed_frames/                                     # Automatically created where transformed images from camera captured images are stored 
├── transormed_images/                                      # Automatically created where transformed images from server downloaded images are stored 
├── venv/                                                   # python virtual environment                      
├── .env                                                    # environment variables
├── .gitignore                                              # Git ignored files
├── classification_results.txt                              # Automatically created for saving strawberry disease classification result
├── main.py                                                 # Entry point for the project
├── README.md                                               # Detailed project description
├── requirements.txt                                        # Required Python dependencies
├── server.py                                               # server configuration using fastapi
```

## Installation
Clone the repository:
```bash
git clone https://github.com/RabindraManandhar/Strawberry_Health_AI
cd Strawberry_Health_AI
```

## Requirements
The project requires the following dependencies:
- python>=3.12.8
- ultralytics>=8.0.0
- torch>=2.4.1
- torchaudio>=2.4.1
- torchvision>=0.19.1
- python-dotenv>=1.0.1
- pandas>=2.2.3
- numpy>=1.26.4
- PyYAML>=6.0.2
- requests>=2.32.3
- opencv-python>=4.10.0.84
- fastapi>=0.115.4
- uvicorn>=0.32.0
- scikit-learn>=1.5.2

## Setup
#### 1. Clone the Repository:
```bash
https://github.com/RabindraManandhar/Strawberry_Health_AI
cd Strawberry_Health_AI
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Set Up Environment Variables<br>
Create a .env file in the root directory with the following:
```bash
STRAWBERRY_DISEASE_KEY=<YOUR_API_KEY>
```

#### 4. Configure Paths:<br>
Update config/config.yaml in root directory with your directories, URLs, and model parameters.

## Usage
#### 1. Run Program:
To run the program
```bash
python main.py
```

#### 2. Download Dataset:
To download and extract the dataset:
```bash
python download_datasets.py
```

#### 3. Train Model:
Fine-tune the YOLOv8 model on the downloaded dataset
```bash
python train_model.py
```

#### 4. Validate Model:
Validate the trained model and save metrics:
```bash
python validate_model.py
```

#### 5. Capture Transform and Classify in Real-Time:
Run the camera-based real-time detection, transformation and classification:
```bash
python camera_image_capture_track_transform_classify.py
```

#### 6. Download Images From Server:
To download images from server:
```bash
python download_images_from_server.py
```

#### 7. Transform and Classify Server Images:
Run the server-based images transformation and classification:
```bash
python server_iamge_transform_classify.py
```

#### 8. Server Operations
Start the server:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration
The configuration file config/config.yaml includes paths, parameters, and hyperparametrs.

## API Endpoints
#### 1. Upload Image
Upload a new image to the server:
- **Method**: POST
- **Endpoint**: /upload_image

#### 2. List Images
List all available images on the server:
- **Method**: GET
- **Endpoint**: /list_images

#### 3. Download Image
Download a specific image from the server:
- **Method**: GET
- **Endpoint**: /get_image/{filename}

#### 4. Download All Images
Download all images from the server as a ZIP file and extract them
- **Method**: GET
- **Endpoint**: /get_all_images


## Acknowledgements
- `YOLOv8` by Ultralytics for model training, validation and inference
- `FastAPI` for server setup
- `OpenCV` for image processing
- `Espressif` for esp32-cam configuration
- `ChatGPT` for troubleshooting