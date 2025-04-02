# Activity Detection Model

A deep learning model for detecting worker productivity states using accelerometer data. This model classifies worker activities into four categories: value-adding work, non-value-adding work, walking, and idling.

## Overview

The productivity state detection module identifies worker productivity states based on accelerometer data from wearable devices. The states are defined as follows:

- **Value-adding work**: Activities directly leading to task completion (e.g., painting a wall)
- **Non-value-adding work**: Activities indirectly supporting task completion (e.g., mixing paint)
- **Walking**: Worker movement/travel
- **Idling**: Non-working states (e.g., resting, talking)

## Project Structure

```
Activity-Detection-Model/
├── data/                    # Dataset directory
│   └── labelled_dataset.csv # Input dataset
├── src/                     # Source code
│   ├── config/             # Configuration module
│   │   ├── __init__.py
│   │   └── config.py       # Model and training parameters
│   ├── data/               # Data handling module
│   │   ├── __init__.py
│   │   └── data_loader.py  # Data preprocessing and loading
│   ├── models/             # Model architecture module
│   │   ├── __init__.py
│   │   └── cnn_model.py    # CNN model implementation
│   └── train.py            # Main training script
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Features

- Real-time activity classification using accelerometer data
- CNN-based deep learning model
- Automatic data preprocessing and balancing
- Early stopping to prevent overfitting
- Model checkpointing for best model preservation
- Comprehensive evaluation metrics and visualizations

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Activity-Detection-Model.git
cd Activity-Detection-Model
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset file (`labelled_dataset.csv`) in the `data/` directory. The dataset should contain the following columns:

   - timestamp
   - acc_x_h, acc_y_h, acc_z_h (head accelerometer data)
   - acc_x_w, acc_y_w, acc_z_w (waist accelerometer data)
   - acc_x_c, acc_y_c, acc_z_c (chest accelerometer data)
   - activity (activity label)

2. Run the training script:

```bash
python src/train.py
```

The script will:

- Load and preprocess the data
- Train the CNN model
- Generate training history plots
- Create a confusion matrix
- Save the trained model and scaler

## Model Architecture

The model uses a CNN architecture with:

- Three convolutional layers (32 filters each)
- Dropout layers (0.2 rate)
- Two dense layers (64 and 4 units)
- Softmax activation for classification