# Sign Language Recognition System

## Overview

This project uses a Convolutional Neural Network (CNN) and computer vision to recognize sign language gestures from images. It explores how deep learning can help make communication more accessible for people with hearing or speech impairments.

## Project Objectives

- Recognize hand gestures from input images.
- Train and test a CNN-based image classification model.
- Explore the use of computer vision for accessible communication.

## Repository Contents

| File or folder | Purpose |
|---|---|
| `app.py` | Python application for using the project. |
| `app.ipynb` | Notebook associated with the application or experimentation. |
| `dataCollection.ipynb` | Notebook for collecting or preparing image data. |
| `augmentData.ipynb` | Notebook for image data augmentation. |
| `buildModel.ipynb` | Notebook for building and training the CNN model. |
| `testModel.ipynb` | Notebook for testing or evaluating the trained model. |
| `sign_language_model.h5` | Saved trained model. |
| `Image/` | Folder containing project images. |

*The descriptions above are based on the filenames; update them if a file serves a different purpose.*

## How It Works

1. Prepare the sign language gesture images.
2. Apply image augmentation where appropriate to support model training.
3. Build and train a Convolutional Neural Network (CNN).
4. Save the trained model in `sign_language_model.h5`.
5. Use the application or testing notebook to make predictions and evaluate the model.

## Technologies

- Python
- Convolutional Neural Networks (CNN)
- Deep Learning
- Computer Vision
- Jupyter Notebook
- HDF5 model format (`.h5`)

## Model Results

The project was previously described as achieving **85% accuracy** on a dataset of **1,200+ images**, with predictions taking under **2 seconds**. Keep these figures here only if they match the results recorded in your notebooks and tests.

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Sign-Language-Recognition-system
```

### 2. Set up Python

Create and activate a virtual environment if desired:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\\Scripts\\activate
```

### 3. Install dependencies

Install the packages required by the project. The exact dependencies should be taken from the imports in `app.py` and the notebooks. If you have a `requirements.txt`, install them with:

```bash
pip install -r requirements.txt
```

### 4. Run the project

If `app.py` is the application entry point, run:

```bash
python app.py
```

You can also open the notebooks in Jupyter Notebook or JupyterLab to review the data preparation, augmentation, model-building, and testing workflow.

## Future Improvements

- Expand the range of gestures the model can recognize.
- Evaluate performance on new images and varied lighting or backgrounds.
- Explore real-time recognition if it is supported by the application.

## Author

Ruchi Kumari
