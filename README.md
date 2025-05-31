# Pothole Detection with YOLOv8

This project uses the YOLOv8 model to detect potholes in images. The `Pothole_detection.ipynb` Jupyter Notebook trains, validates, and predicts pothole locations using a dataset from Roboflow, leveraging the `ultralytics` library for YOLOv8 and `roboflow` for dataset management.

## Project Overview
- **Objective**: Train a YOLOv8 model to detect potholes in images for road safety applications.
- **Dataset**: Pothole detection dataset (version 16) from Roboflow workspace `jerry-cooper-tlzkx`, project `pothole_detection-hfnqo`.
- **Model**: YOLOv8s (small variant), fine-tuned for 5 epochs.
- **Environment**: GPU-enabled (e.g., Google Colab with T4 GPU).
- **Outputs**: Trained model weights, validation metrics, and predicted images with bounding boxes.

## Prerequisites
- Python 3.11.3 or compatible.
- GPU (recommended for training; e.g., NVIDIA T4 in Colab).
- Roboflow account and API key (sign up at https://app.roboflow.com).
- Git (optional, for cloning the repository).

## Setup Instructions
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd pothole-detection
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set Up Roboflow API Key**:
   - Obtain your API key from https://app.roboflow.com/settings/api.
   - Set it as an environment variable (recommended):
     ```bash
     export ROBOFLOW_API_KEY=your_api_key_here  # On Windows: set ROBOFLOW_API_KEY=your_api_key_here
     ```
   - Alternatively, replace `api_key=""` in the notebook’s Roboflow cell with your key (less secure).

5. **Prepare the Notebook**:
   - Open `Pothole_detection.ipynb` in Jupyter Notebook or Google Colab.
   - If using Colab, ensure a GPU runtime (`Runtime > Change runtime type > GPU`).

## Running the Notebook
1. **Check GPU Availability**:
   - Run the first cell (`!nvidia-smi`) to confirm GPU access.

2. **Download Dataset**:
   - Run the cell with `roboflow` code to download the dataset to `{HOME}/datasets`.
   - Ensure your Roboflow API key is set.

3. **Train the Model**:
   - Run the training cell (`!yolo task=detect mode=train ...`) to train YOLOv8s for 5 epochs.
   - Outputs are saved in `{HOME}/runs/detect/train`.

4. **Validate the Model**:
   - Run the validation cell to evaluate the model on the validation set.
   - Metrics are saved in `{HOME}/runs/detect/val`.

5. **Predict on Test Images**:
   - Run the prediction cell to detect potholes in test images.
   - Results are saved in `{HOME}/runs/detect/predict`.

6. **View Results**:
   - Run the final cell to display predicted images with bounding boxes.
   - Images are loaded from `runs/detect/predict/*.jpg`.

## Expected Outputs
- **Training**: Model weights (`best.pt`, `last.pt`) in `{HOME}/runs/detect/train/weights`.
- **Validation**: Metrics (e.g., mAP, precision, recall) in `{HOME}/runs/detect/val`.
- **Prediction**: Images with bounding boxes in `{HOME}/runs/detect/predict`.
- **Notebook Outputs**: Displayed images and logs in the notebook.

## Troubleshooting
- **Roboflow API Key Error**:
  - Ensure the API key is set correctly.
  - Verify your Roboflow account has access to the `jerry-cooper-tlzkx/pothole_detection-hfnqo` project.
- **GPU Not Available**:
  - In Colab, switch to a GPU runtime.
  - Locally, ensure an NVIDIA GPU and CUDA are installed.
- **Dataset Download Fails**:
  - Check internet connectivity.
  - Confirm the dataset version (16) is still available on Roboflow.
- **Module Not Found**:
  - Reinstall dependencies: `pip install -r requirements.txt`.
  - Ensure `ultralytics==8.0.196` is installed.

## Security Notes
- **Roboflow API Key**:
  - Do not hardcode your API key in the notebook.
  - Use environment variables or a `.env` file.
  - Clear notebook outputs before sharing to avoid leaking keys.
- **Notebook Outputs**:
  - Outputs may contain file paths or logs. Review before sharing.

## Dataset
- **Source**: Roboflow (`jerry-cooper-tlzkx/pothole_detection-hfnqo`, version 16).
- **Format**: YOLOv8-compatible (includes `data.yaml`, train/valid/test splits).
- **Access**: Requires a Roboflow account and API key.
- **Details**: Images with pothole annotations (bounding boxes).

## Contributing
- Report issues or suggest improvements via GitHub Issues.
- Submit pull requests for bug fixes or enhancements.


## Acknowledgments
- **Ultralytics**: For the YOLOv8 implementation.
- **Roboflow**: For providing the pothole detection dataset.
- **Google Colab**: For free GPU access.

## tutorial link 
https://www.linkedin.com/posts/kandregula-prem-kumar-059642238_saisatish-indianservers-aimers-activity-7218276378629005312-aK8Y?utm_source=share&utm_medium=member_desktop&rcm=ACoAADsu2cUBj_2awdINrbYALkhoaUcjqmPZDFs
