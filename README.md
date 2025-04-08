# Cataract Detection App

**Repository:** [rajkadakia/cataract-detection-app](https://github.com/rajkadakia/cataract-detection-app)

This project is a cataract detection system built using PyTorch and Streamlit. It includes model training, Grad-CAM visualization, and a user-friendly web interface for single image classification.

## Features
- PyTorch-based deep learning model
- Grad-CAM for visual explanations
- Streamlit app for easy predictions
- Confidence score for predictions
- Progress bar and clean UI

## Files
- `modeltraining.ipynb` : Model training notebook
- `detection_app.py` : Streamlit web application
- `cataract_model.pth` : Trained model file
- `processed_images/` : Dataset used for training and testing

## Run the App
```bash
streamlit run detection_app.py
